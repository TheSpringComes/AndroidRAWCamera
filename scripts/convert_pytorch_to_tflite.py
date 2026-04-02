#!/usr/bin/env python3
"""
PyTorch 模型 → TensorFlow Lite 转换脚本

将你的 PyTorch (.pt / .ptl) 模型转换为 TFLite (.tflite) 格式，
以便在 Android 上通过 NNAPI 调用 NPU 加速推理。

转换路径: PyTorch → ONNX → TensorFlow → TFLite

使用方法:
    pip install torch onnx onnx-tf tensorflow tf2onnx
    python convert_pytorch_to_tflite.py \
        --model_path  your_model.pt \
        --input_shape 1,1,3000,4000 \
        --output_path model.tflite

转换完成后将 model.tflite 放到:
    Application/src/main/assets/model.tflite
"""

import argparse
import os
import sys

import numpy as np
import torch
import onnx


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch → TFLite 模型转换工具")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="PyTorch 模型路径 (.pt / .pth / .ptl)"
    )
    parser.add_argument(
        "--input_shape", type=str, default="1,1,3000,4000",
        help="输入张量形状，逗号分隔 (默认: 1,1,3000,4000 即 batch,channels,height,width)"
    )
    parser.add_argument(
        "--output_path", type=str, default="model.tflite",
        help="输出 TFLite 模型路径 (默认: model.tflite)"
    )
    parser.add_argument(
        "--quantize", action="store_true",
        help="启用 float16 量化（减小模型体积，NPU 友好）"
    )
    return parser.parse_args()


def load_pytorch_model(model_path):
    """加载 PyTorch JIT 模型"""
    print(f"[1/4] 加载 PyTorch 模型: {model_path}")
    if model_path.endswith(".ptl"):
        model = torch.jit.load(model_path, map_location="cpu")
    else:
        model = torch.load(model_path, map_location="cpu")
    model.eval()
    return model


def export_to_onnx(model, input_shape, onnx_path):
    """PyTorch → ONNX"""
    print(f"[2/4] 导出 ONNX: {onnx_path}")
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
    )
    # 验证 ONNX 模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"   ONNX 模型验证通过")


def convert_onnx_to_tflite(onnx_path, output_path, input_shape, quantize=False):
    """ONNX → TFLite (通过 tf2onnx 或 onnx-tf)"""
    print(f"[3/4] ONNX → TensorFlow SavedModel")

    try:
        # 方式 1: 使用 onnx-tf
        from onnx_tf.backend import prepare
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        saved_model_dir = output_path.replace(".tflite", "_saved_model")
        tf_rep.export_graph(saved_model_dir)
    except ImportError:
        # 方式 2: 使用 tf2onnx (反向，从 ONNX 直接转 TFLite)
        print("   onnx-tf 未安装，尝试 tf2onnx 路径...")
        import subprocess
        saved_model_dir = output_path.replace(".tflite", "_saved_model")
        subprocess.run([
            sys.executable, "-m", "tf2onnx.convert",
            "--onnx", onnx_path,
            "--output", saved_model_dir,
            "--opset", "13"
        ], check=True)

    print(f"[4/4] TensorFlow SavedModel → TFLite: {output_path}")
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # 优化选项
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if quantize:
        # float16 量化 — NPU 友好，精度损失小
        converter.target_spec.supported_types = [tf.float16]
        print("   启用 float16 量化")

    # 允许 TFLite 内置算子 + Select TF 算子（提升兼容性）
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n转换完成！")
    print(f"   输出文件: {output_path}")
    print(f"   文件大小: {size_mb:.1f} MB")
    print(f"\n下一步:")
    print(f"   cp {output_path} Application/src/main/assets/model.tflite")


def verify_tflite(output_path, input_shape):
    """验证 TFLite 模型可正常推理"""
    print(f"\n验证 TFLite 模型...")
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"   输入: {input_details[0]['shape']} {input_details[0]['dtype']}")
    print(f"   输出: {output_details[0]['shape']} {output_details[0]['dtype']}")

    # 用随机数据测试推理
    test_input = np.random.rand(*input_shape).astype(np.float32)
    interpreter.resize_tensor_input(input_details[0]['index'], list(input_shape))
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"   推理输出形状: {output.shape}")
    print(f"   验证通过！模型可正常运行。")


def main():
    args = parse_args()
    input_shape = tuple(int(x) for x in args.input_shape.split(","))
    onnx_path = args.output_path.replace(".tflite", ".onnx")

    model = load_pytorch_model(args.model_path)
    export_to_onnx(model, input_shape, onnx_path)
    convert_onnx_to_tflite(onnx_path, args.output_path, input_shape, args.quantize)
    verify_tflite(args.output_path, input_shape)

    # 清理中间文件
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
        print(f"\n已清理中间文件: {onnx_path}")


if __name__ == "__main__":
    main()
