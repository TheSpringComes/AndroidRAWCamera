package com.example.android.camera2basic;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * NPU 推理辅助类，封装 TensorFlow Lite + NNAPI Delegate。
 *
 * 加速策略优先级：
 * 1. NNAPI Delegate → 委托给手机 NPU/DSP（需 Android 8.1+）
 * 2. GPU Delegate  → 委托给 GPU（兜底加速）
 * 3. CPU           → 纯 CPU 多线程推理（最终兜底）
 *
 * 使用方式：
 *   NpuModelHelper helper = new NpuModelHelper(context, "model.tflite");
 *   float[] output = helper.runInference(inputFloat, 1, 1, height, width, outputSize);
 *   helper.close();
 */
public class NpuModelHelper {

    private static final String TAG = "NpuModelHelper";

    public enum AcceleratorType {
        NNAPI,  // NPU / DSP
        GPU,
        CPU
    }

    private Interpreter interpreter;
    private NnApiDelegate nnApiDelegate;
    private GpuDelegate gpuDelegate;
    private AcceleratorType activeAccelerator = AcceleratorType.CPU;

    /**
     * 初始化 TFLite 解释器，优先尝试 NNAPI（NPU），失败则降级到 GPU，最后 CPU。
     *
     * @param context   应用上下文
     * @param modelName assets 目录中的 .tflite 模型文件名
     */
    public NpuModelHelper(Context context, String modelName) throws IOException {
        MappedByteBuffer modelBuffer = loadModelFile(context, modelName);

        // 优先尝试 NNAPI (NPU/DSP)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            try {
                NnApiDelegate.Options nnApiOptions = new NnApiDelegate.Options();
                // 允许 NNAPI 选择最佳加速器（NPU > DSP > GPU）
                nnApiOptions.setUseNnapiCpu(false);
                // 如果需要指定特定加速器，可以取消注释：
                // nnApiOptions.setAcceleratorName("google-edgetpu");  // Pixel NPU
                // nnApiOptions.setAcceleratorName("qti-hta");         // Qualcomm HTA (NPU)
                // nnApiOptions.setAcceleratorName("qti-dsp");         // Qualcomm DSP
                // nnApiOptions.setAcceleratorName("mtk-neuron");      // MediaTek APU
                // nnApiOptions.setAcceleratorName("samsung-eden");    // Samsung NPU
                nnApiDelegate = new NnApiDelegate(nnApiOptions);

                Interpreter.Options options = new Interpreter.Options();
                options.addDelegate(nnApiDelegate);
                options.setNumThreads(4);
                interpreter = new Interpreter(modelBuffer, options);
                activeAccelerator = AcceleratorType.NNAPI;
                Log.i(TAG, "NNAPI delegate 初始化成功，使用 NPU/DSP 加速");
                return;
            } catch (Exception e) {
                Log.w(TAG, "NNAPI delegate 初始化失败，尝试 GPU: " + e.getMessage());
                if (nnApiDelegate != null) {
                    nnApiDelegate.close();
                    nnApiDelegate = null;
                }
            }
        }

        // 降级到 GPU Delegate
        try {
            GpuDelegate.Options gpuOptions = new GpuDelegate.Options();
            gpuDelegate = new GpuDelegate(gpuOptions);

            Interpreter.Options options = new Interpreter.Options();
            options.addDelegate(gpuDelegate);
            options.setNumThreads(4);
            interpreter = new Interpreter(modelBuffer, options);
            activeAccelerator = AcceleratorType.GPU;
            Log.i(TAG, "GPU delegate 初始化成功");
            return;
        } catch (Exception e) {
            Log.w(TAG, "GPU delegate 初始化失败，降级到 CPU: " + e.getMessage());
            if (gpuDelegate != null) {
                gpuDelegate.close();
                gpuDelegate = null;
            }
        }

        // 最终兜底：CPU
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        interpreter = new Interpreter(modelBuffer, options);
        activeAccelerator = AcceleratorType.CPU;
        Log.i(TAG, "使用 CPU 推理 (4 线程)");
    }

    /**
     * 执行推理。
     *
     * @param inputData  输入浮点数组（归一化后的像素数据）
     * @param batch      批次大小（通常为 1）
     * @param channels   输入通道数（RAW 数据为 1）
     * @param height     图像高度
     * @param width      图像宽度
     * @param outputSize 输出数组总大小（如 3 * height * width）
     * @return 输出浮点数组
     */
    public float[] runInference(float[] inputData, int batch, int channels,
                                int height, int width, int outputSize) {
        // 准备输入 ByteBuffer（TFLite 要求 ByteBuffer 输入）
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(batch * channels * height * width * 4);
        inputBuffer.order(ByteOrder.nativeOrder());
        for (float val : inputData) {
            inputBuffer.putFloat(val);
        }
        inputBuffer.rewind();

        // 准备输出 ByteBuffer
        ByteBuffer outputBuffer = ByteBuffer.allocateDirect(outputSize * 4);
        outputBuffer.order(ByteOrder.nativeOrder());

        // 推理
        interpreter.run(inputBuffer, outputBuffer);

        // 解析输出
        outputBuffer.rewind();
        float[] output = new float[outputSize];
        outputBuffer.asFloatBuffer().get(output);
        return output;
    }

    /**
     * 获取当前使用的加速器类型
     */
    public AcceleratorType getActiveAccelerator() {
        return activeAccelerator;
    }

    /**
     * 获取加速器的中文描述
     */
    public String getAcceleratorName() {
        switch (activeAccelerator) {
            case NNAPI: return "NPU/DSP (NNAPI)";
            case GPU:   return "GPU";
            case CPU:   return "CPU";
            default:    return "Unknown";
        }
    }

    /**
     * 释放资源
     */
    public void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
        if (nnApiDelegate != null) {
            nnApiDelegate.close();
            nnApiDelegate = null;
        }
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }
    }

    /**
     * 从 assets 加载模型文件为 MappedByteBuffer
     */
    private MappedByteBuffer loadModelFile(Context context, String modelName) throws IOException {
        // 先将 asset 复制到 filesDir，再用 FileChannel 映射
        File file = new File(context.getFilesDir(), modelName);
        if (!file.exists()) {
            try (InputStream is = context.getAssets().open(modelName);
                 FileOutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int r;
                while ((r = is.read(buffer)) != -1) {
                    os.write(buffer, 0, r);
                }
            }
        }
        FileInputStream fis = new FileInputStream(file);
        FileChannel fileChannel = fis.getChannel();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, file.length());
    }
}
