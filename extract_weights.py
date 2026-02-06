#!/usr/bin/env python3
"""
Extract quantized weights from braggnn_int8_qlinear_opset10.onnx
and output them as a C header file (braggnn.h) for Gemmini.

- QLinearConv layers: INT8 weights and INT32 biases extracted directly
- Gemm (FP32) layers: weights quantized to INT8 using surrounding
  DequantizeLinear/QuantizeLinear scales
- Conv weights transposed from ONNX [N,C,H,W] to Gemmini [N,H,W,C]
- 1x1 conv weights reshaped from [N,C,1,1] to [N,C]
- FC weights kept as [out_features, in_features]
"""

import numpy as np
import onnx
from onnx import numpy_helper
import sys
import os

# Import scale extraction utilities from existing script
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
_scales_mod = import_module("extract-onnx-scales")
extract_scales = _scales_mod.extract_scales
extract_weights_fp32 = _scales_mod.extract_weights
calculate_weight_scale = _scales_mod.calculate_weight_scale
calculate_acc_scales = _scales_mod.calculate_acc_scales
calculate_fc_scales = _scales_mod.calculate_fc_scales


def load_model(model_path):
    model = onnx.load(model_path)
    initializers = {}
    for init in model.graph.initializer:
        initializers[init.name] = numpy_helper.to_array(init)
    return model, initializers


def get_scale(initializers, name):
    return float(initializers[name])


def quantize_weights_symmetric(fp32_weights, w_scale=None):
    """Quantize FP32 weights to INT8 using symmetric quantization."""
    if w_scale is None:
        w_scale = calculate_weight_scale(fp32_weights)
    int8_weights = np.clip(np.round(fp32_weights / w_scale), -128, 127).astype(np.int8)
    return int8_weights, w_scale


def quantize_bias(fp32_bias, x_scale, w_scale):
    """Quantize FP32 bias to INT32: bias_q = round(bias_fp32 / (x_scale * w_scale))"""
    bias_scale = x_scale * w_scale
    int32_bias = np.round(fp32_bias / bias_scale).astype(np.int32)
    return int32_bias


def format_1d_array(arr, per_line=8, indent="    "):
    """Format a 1D array as C initializer."""
    lines = []
    for i in range(0, len(arr), per_line):
        chunk = arr[i:i+per_line]
        vals = ", ".join(f"{int(v):>6d}" for v in chunk)
        lines.append(f"{indent} {vals}")
    return ",\n".join(lines)


def format_conv_weights_okkc(weights_nchw):
    """Format conv weights from ONNX [N,C,H,W] to C array [N][H][W][C]."""
    # Transpose: [N,C,H,W] -> [N,H,W,C]
    weights_nhwc = np.transpose(weights_nchw, (0, 2, 3, 1))
    N, H, W, C = weights_nhwc.shape
    lines = []
    for n in range(N):
        filter_lines = []
        for h in range(H):
            row_lines = []
            for w_idx in range(W):
                vals = ", ".join(f"{int(v):>4d}" for v in weights_nhwc[n, h, w_idx])
                row_lines.append(f"{{{vals}}}")
            filter_lines.append("{" + ", ".join(row_lines) + "}")
        lines.append("         {" + ", ".join(filter_lines) + "}")
    return ",\n".join(lines)


def format_1x1_conv_weights(weights_nc11):
    """Format 1x1 conv weights from ONNX [N,C,1,1] to C array [N][C]."""
    N, C = weights_nc11.shape[0], weights_nc11.shape[1]
    weights_2d = weights_nc11.reshape(N, C)
    lines = []
    for n in range(N):
        vals = ", ".join(f"{int(v):>4d}" for v in weights_2d[n])
        lines.append(f"        {{{vals}}}")
    return ",\n".join(lines)


def format_fc_weights(weights_oi):
    """Format FC weights as C array [out][in]."""
    O, I = weights_oi.shape
    lines = []
    for o in range(O):
        vals = ", ".join(f"{int(v):>4d}" for v in weights_oi[o])
        lines.append(f"    {{{vals}}}")
    return ",\n".join(lines)


def main():
    model_path = os.path.join(os.path.dirname(__file__),
                              "onnx/braggnn_int8_qlinear_opset10.onnx")
    output_path = os.path.join(os.path.dirname(__file__), "braggnn.h")

    print(f"Loading model: {model_path}")
    model, init = load_model(model_path)

    # =====================================================================
    # 1. Extract Conv layer weights (already INT8 in QLinearConv)
    # =====================================================================

    # Conv1: [64, 1, 3, 3] -> [64][3][3][1]
    conv1_w = init["cnn_layers.0.weight_quantized"]
    conv1_b = init["cnn_layers.0.bias_quantized"]
    conv1_w_scale = get_scale(init, "cnn_layers.0.weight_scale")
    input_scale = get_scale(init, "input_scale")
    conv1_y_scale = get_scale(init, "/cnn_layers.0/Conv_output_0_scale")

    # Conv2: [32, 64, 3, 3] -> [32][3][3][64]
    conv2_w = init["cnn_layers.2.weight_quantized"]
    conv2_b = init["cnn_layers.2.bias_quantized"]
    conv2_w_scale = get_scale(init, "cnn_layers.2.weight_scale")
    conv2_x_scale = get_scale(init, "/cnn_layers.1/LeakyRelu_output_0_scale")
    conv2_y_scale = get_scale(init, "/cnn_layers.2/Conv_output_0_scale")

    # Conv3: [8, 32, 3, 3] -> [8][3][3][32]
    conv3_w = init["cnn_layers.4.weight_quantized"]
    conv3_b = init["cnn_layers.4.bias_quantized"]
    conv3_w_scale = get_scale(init, "cnn_layers.4.weight_scale")
    conv3_x_scale = get_scale(init, "/cnn_layers.3/LeakyRelu_output_0_scale")
    conv3_y_scale = get_scale(init, "/cnn_layers.4/Conv_output_0_scale")

    # =====================================================================
    # 2. Extract NLB 1x1 conv weights (already INT8 in QLinearConv)
    # =====================================================================

    nlb_theta_w = init["nlb.theta_layer.weight_quantized"]  # [32, 64, 1, 1]
    nlb_theta_b = init["nlb.theta_layer.bias_quantized"]
    nlb_theta_w_scale = get_scale(init, "nlb.theta_layer.weight_scale")
    nlb_theta_y_scale = get_scale(init, "/nlb/theta_layer/Conv_output_0_scale")

    nlb_phi_w = init["nlb.phi_layer.weight_quantized"]  # [32, 64, 1, 1]
    nlb_phi_b = init["nlb.phi_layer.bias_quantized"]
    nlb_phi_w_scale = get_scale(init, "nlb.phi_layer.weight_scale")
    nlb_phi_y_scale = get_scale(init, "/nlb/phi_layer/Conv_output_0_scale")

    nlb_g_w = init["nlb.g_layer.weight_quantized"]  # [32, 64, 1, 1]
    nlb_g_b = init["nlb.g_layer.bias_quantized"]
    nlb_g_w_scale = get_scale(init, "nlb.g_layer.weight_scale")
    nlb_g_y_scale = get_scale(init, "/nlb/g_layer/Conv_output_0_scale")

    nlb_out_w = init["nlb.out_cnn.weight_quantized"]  # [64, 32, 1, 1]
    nlb_out_b = init["nlb.out_cnn.bias_quantized"]
    nlb_out_w_scale = get_scale(init, "nlb.out_cnn.weight_scale")
    nlb_out_x_scale = get_scale(init, "/nlb/Reshape_3_output_0_scale")
    nlb_out_y_scale = get_scale(init, "/nlb/out_cnn/Conv_output_0_scale")

    # NLB Add scales
    nlb_add_a_scale = nlb_out_y_scale  # out_cnn output
    nlb_add_b_scale = conv1_y_scale    # skip connection (conv1 output)
    nlb_add_y_scale = get_scale(init, "/nlb/Add_output_0_scale")

    # NLB MatMul scales
    nlb_matmul_x_scale = nlb_theta_y_scale  # theta output (transposed)
    nlb_matmul_w_scale = nlb_phi_y_scale    # phi output (reshaped)
    nlb_matmul_y_scale = get_scale(init, "/nlb/MatMul_output_0_scale")

    softmax_y_scale = get_scale(init, "/nlb/atten_act/Softmax_output_0_scale")

    nlb_matmul1_x_scale = softmax_y_scale
    nlb_matmul1_w_scale = nlb_g_y_scale  # g output (transposed)
    nlb_matmul1_y_scale = get_scale(init, "/nlb/MatMul_1_output_0_scale")

    # LeakyRelu scales
    leakyrelu1_x_scale = nlb_add_y_scale
    leakyrelu1_y_scale = get_scale(init, "/cnn_layers.1/LeakyRelu_output_0_scale")
    leakyrelu3_x_scale = conv2_y_scale
    leakyrelu3_y_scale = get_scale(init, "/cnn_layers.3/LeakyRelu_output_0_scale")
    leakyrelu5_x_scale = conv3_y_scale
    leakyrelu5_y_scale = get_scale(init, "/cnn_layers.5/LeakyRelu_output_0_scale")

    # =====================================================================
    # 3. Quantize FC layer weights (FP32 Gemm -> INT8)
    #    Pattern: DequantizeLinear -> Gemm(FP32) -> QuantizeLinear
    #    w_scale = max(|w|) / 127
    #    int8_w = round(fp32_w / w_scale)
    #    int32_b = round(fp32_b / (x_scale * w_scale))
    # =====================================================================

    fc_layers = [
        {
            "name": "dense_layers.0",
            "w_key": "dense_layers.0.weight",     # [16, 200]
            "b_key": "dense_layers.0.bias",
            "x_scale": leakyrelu5_y_scale,  # cnn_layers.5/LeakyRelu output
            "y_scale": get_scale(init, "/dense_layers.0/Gemm_output_0_scale"),
        },
        {
            "name": "dense_layers.2",
            "w_key": "dense_layers.2.weight",     # [8, 16]
            "b_key": "dense_layers.2.bias",
            "x_scale": get_scale(init, "/dense_layers.1/LeakyRelu_output_0_scale"),
            "y_scale": get_scale(init, "/dense_layers.2/Gemm_output_0_scale"),
        },
        {
            "name": "dense_layers.4",
            "w_key": "dense_layers.4.weight",     # [4, 8]
            "b_key": "dense_layers.4.bias",
            "x_scale": get_scale(init, "/dense_layers.3/LeakyRelu_output_0_scale"),
            "y_scale": get_scale(init, "/dense_layers.4/Gemm_output_0_scale"),
        },
        {
            "name": "dense_layers.6",
            "w_key": "dense_layers.6.weight",     # [2, 4]
            "b_key": "dense_layers.6.bias",
            "x_scale": get_scale(init, "/dense_layers.5/LeakyRelu_output_0_scale"),
            "y_scale": get_scale(init, "/dense_layers.6/Gemm_output_0_scale"),
        },
        {
            "name": "dense_layers.8",
            "w_key": "dense_layers.8.weight",     # [2, 2]
            "b_key": "dense_layers.8.bias",
            "x_scale": get_scale(init, "/dense_layers.7/LeakyRelu_output_0_scale"),
            "y_scale": input_scale,  # output mapped to same scale as input (1/127)
        },
    ]

    fc_results = []
    for fc in fc_layers:
        fp32_w = init[fc["w_key"]]
        fp32_b = init[fc["b_key"]]
        int8_w, w_scale = quantize_weights_symmetric(fp32_w)
        int32_b = quantize_bias(fp32_b, fc["x_scale"], w_scale)
        acc_scale = fc["x_scale"] * w_scale / fc["y_scale"]
        fc_results.append({
            "name": fc["name"],
            "int8_w": int8_w,
            "int32_b": int32_b,
            "w_scale": w_scale,
            "x_scale": fc["x_scale"],
            "y_scale": fc["y_scale"],
            "acc_scale": acc_scale,
            "fp32_w": fp32_w,
            "fp32_b": fp32_b,
        })

        print(f"  {fc['name']}: w_shape={int8_w.shape}, "
              f"w_scale={w_scale:.10e}, "
              f"x_scale={fc['x_scale']:.10e}, "
              f"y_scale={fc['y_scale']:.10e}, "
              f"acc_scale={acc_scale:.10e}")

    # LeakyRelu scales for dense layers
    dense_leakyrelu_scales = [
        ("dense_layers.1", get_scale(init, "/dense_layers.0/Gemm_output_0_scale"),
         get_scale(init, "/dense_layers.1/LeakyRelu_output_0_scale")),
        ("dense_layers.3", get_scale(init, "/dense_layers.2/Gemm_output_0_scale"),
         get_scale(init, "/dense_layers.3/LeakyRelu_output_0_scale")),
        ("dense_layers.5", get_scale(init, "/dense_layers.4/Gemm_output_0_scale"),
         get_scale(init, "/dense_layers.5/LeakyRelu_output_0_scale")),
        ("dense_layers.7", get_scale(init, "/dense_layers.6/Gemm_output_0_scale"),
         get_scale(init, "/dense_layers.7/LeakyRelu_output_0_scale")),
    ]

    # =====================================================================
    # 4. Generate header file
    # =====================================================================
    print(f"\nWriting header to: {output_path}")

    with open(output_path, "w") as f:
        f.write('#include "include/gemmini_testutils.h"\n\n')
        f.write("\n")
        f.write("// BraggNN Model Architecture Configuration\n")
        f.write("// Input: 11x11 patches, 3 Conv layers + NLB, 4 FC layers\n")
        f.write(f"#define INPUT_DIM 11     // Input patch size (11x11)\n")
        f.write(f"#define INPUT_CHANNELS 1 // Grayscale input\n\n")
        f.write(f"#define BATCH 1\n\n")

        f.write("// Convolutional layers configuration\n")
        f.write("// 3 CNN layers: cnn_out_chs = (64, 32, 8), cnn_in_chs = (1, 64, 32)\n")
        f.write(f"#define CONV1_FILTERS 64 // Conv1: 11x11x1 -> 9x9x64\n")
        f.write(f"#define CONV1_KERNEL 3   // 3x3 kernel\n")
        f.write(f"#define CONV1_DIM 9\n")
        f.write(f"#define CONV1_2D CONV1_DIM *CONV1_DIM\n")
        f.write(f"#define CONV1_CHANNELS 1\n")
        f.write(f"#define CONV2_FILTERS 32 // Conv2: 9x9x64 -> 7x7x32\n")
        f.write(f"#define CONV2_KERNEL 3   // 3x3 kernel\n")
        f.write(f"#define CONV2_DIM 7\n")
        f.write(f"#define CONV2_CHANNELS 1\n")
        f.write(f"#define CONV3_FILTERS 8 // Conv3: 7x7x32 -> 5x5x8\n")
        f.write(f"#define CONV3_KERNEL 3  // 3x3 kernel\n")
        f.write(f"#define CONV3_DIM 5\n")
        f.write(f"#define CONV3_2D CONV3_DIM *CONV3_DIM\n")
        f.write(f"#define CONV3_FLATTENED CONV3_2D *CONV3_FILTERS\n")
        f.write(f"#define CONV3_CHANNELS 1\n\n")

        f.write("// Non-Local Block kernel sizes (all 1x1 convolutions)\n")
        f.write(f"#define NLB_THETA_KERNEL 1  // theta_layer: 1x1 conv\n")
        f.write(f"#define NLB_PHI_KERNEL 1    // phi_layer: 1x1 conv\n")
        f.write(f"#define NLB_G_KERNEL 1      // g_layer: 1x1 conv\n")
        f.write(f"#define NLB_OUT_KERNEL 1    // out_cnn: 1x1 conv\n\n")

        f.write("// Fully connected layers (fcsz=(16, 8, 4, 2))\n")
        f.write(f"#define FC1_UNITS 16   // First FC layer units\n")
        f.write(f"#define FC2_UNITS 8    // Second FC layer units\n")
        f.write(f"#define FC3_UNITS 4    // Third FC layer units\n")
        f.write(f"#define FC4_UNITS 2    // Fourth FC layer units\n")
        f.write(f"#define OUTPUT_UNITS 2 // Output predictions (x, y coordinates of peak center)\n\n")

        # --- Accumulator Scales ---
        f.write("// Accumulator Scales\n\n")

        # Conv layer acc scales
        conv_acc_scales = [
            ("/cnn_layers.0/Conv_quant (QLinearConv)",
             input_scale, conv1_w_scale, conv1_y_scale),
            ("/nlb/theta_layer/Conv_quant (QLinearConv)",
             conv1_y_scale, nlb_theta_w_scale, nlb_theta_y_scale),
            ("/nlb/phi_layer/Conv_quant (QLinearConv)",
             conv1_y_scale, nlb_phi_w_scale, nlb_phi_y_scale),
            ("/nlb/g_layer/Conv_quant (QLinearConv)",
             conv1_y_scale, nlb_g_w_scale, nlb_g_y_scale),
        ]

        scale_names = [
            "CNN_LAYERS_0_CONV_QUANT_ACC_SCALE",
            "NLB_THETA_LAYER_CONV_QUANT_ACC_SCALE",
            "NLB_PHI_LAYER_CONV_QUANT_ACC_SCALE",
            "NLB_G_LAYER_CONV_QUANT_ACC_SCALE",
        ]

        for (desc, xs, ws, ys), sname in zip(conv_acc_scales, scale_names):
            acc = xs * ws / ys
            f.write(f"// {desc}\n")
            f.write(f"// x_scale={xs}, w_scale={ws}, y_scale={ys}\n")
            f.write(f"#define {sname} {acc:.10e}f\n\n")

        # NLB MatMul acc scales
        nlb_mm_acc = nlb_matmul_x_scale * nlb_matmul_w_scale / nlb_matmul_y_scale
        f.write(f"// /nlb/MatMul_quant (QLinearMatMul)\n")
        f.write(f"// x_scale={nlb_matmul_x_scale}, w_scale={nlb_matmul_w_scale}, y_scale={nlb_matmul_y_scale}\n")
        f.write(f"#define NLB_MATMUL_QUANT_ACC_SCALE {nlb_mm_acc:.10e}f\n\n")

        # Softmax scales
        f.write(f"// Softmax scales (for manual softmax implementation)\n")
        f.write(f"#define SOFTMAX_INPUT_SCALE {nlb_matmul_y_scale}f   // matmul output scale\n")
        f.write(f"#define SOFTMAX_OUTPUT_SCALE {softmax_y_scale}f // softmax output scale\n\n")

        # NLB MatMul_1 acc scale
        nlb_mm1_acc = nlb_matmul1_x_scale * nlb_matmul1_w_scale / nlb_matmul1_y_scale
        f.write(f"// /nlb/MatMul_1_quant (QLinearMatMul)\n")
        f.write(f"// x_scale={nlb_matmul1_x_scale}, w_scale={nlb_matmul1_w_scale}, y_scale={nlb_matmul1_y_scale}\n")
        f.write(f"#define NLB_MATMUL_1_QUANT_ACC_SCALE {nlb_mm1_acc:.10e}f\n\n")

        # NLB out_cnn conv acc scale
        nlb_out_acc = nlb_out_x_scale * nlb_out_w_scale / nlb_out_y_scale
        f.write(f"// /nlb/out_cnn/Conv_quant (QLinearConv)\n")
        f.write(f"// x_scale={nlb_out_x_scale}, w_scale={nlb_out_w_scale}, y_scale={nlb_out_y_scale}\n")
        f.write(f"#define NLB_OUT_CNN_CONV_QUANT_ACC_SCALE {nlb_out_acc:.10e}f\n\n")

        # NLB Add scales
        nlb_add_a_ratio = nlb_add_a_scale / nlb_add_y_scale
        nlb_add_b_ratio = nlb_add_b_scale / nlb_add_y_scale
        f.write(f"// /nlb/Add_quant (QLinearAdd)\n")
        f.write(f"// input_a (out_cnn): scale={nlb_add_a_scale}\n")
        f.write(f"// input_b (conv1/skip): scale={nlb_add_b_scale}\n")
        f.write(f"// output: scale={nlb_add_y_scale}\n")
        f.write(f"// For tiled_resadd_auto: output = (A * A_scale + B * B_scale) * C_scale\n")
        f.write(f"// A_scale = out_cnn_scale / output_scale = {nlb_add_a_scale} / {nlb_add_y_scale} = {nlb_add_a_ratio:.3f}\n")
        f.write(f"// B_scale = conv1_scale / output_scale = {nlb_add_b_scale} / {nlb_add_y_scale} = {nlb_add_b_ratio:.3f}\n")
        f.write(f"// C_scale = 1.0 (ACC_SCALE_IDENTITY)\n")
        f.write(f"#define NLB_ADD_A_SCALE {nlb_add_a_ratio:.10e}f  // nlb_output (out_cnn) input scale\n")
        f.write(f"#define NLB_ADD_B_SCALE {nlb_add_b_ratio:.10e}f  // skip connection (conv1) input scale\n")
        f.write(f"// Use ACC_SCALE_IDENTITY (1.0) for C_scale\n\n")

        # LeakyRelu acc scales for CNN layers
        cnn_leakyrelu_data = [
            ("/cnn_layers.1/LeakyRelu_quant", leakyrelu1_x_scale, leakyrelu1_y_scale,
             "CNN_LAYERS_1_LEAKYRELU_QUANT_ACC_SCALE"),
            ("/cnn_layers.3/LeakyRelu_quant", leakyrelu3_x_scale, leakyrelu3_y_scale,
             "CNN_LAYERS_3_LEAKYRELU_QUANT_ACC_SCALE"),
            ("/cnn_layers.5/LeakyRelu_quant", leakyrelu5_x_scale, leakyrelu5_y_scale,
             "CNN_LAYERS_5_LEAKYRELU_QUANT_ACC_SCALE"),
        ]
        for desc, xs, ys, sname in cnn_leakyrelu_data:
            acc = xs / ys
            f.write(f"// {desc} (QLinearLeakyRelu)\n")
            f.write(f"// x_scale={xs}, w_scale=None, y_scale={ys}\n")
            f.write(f"#define {sname} {acc:.10e}f\n\n")

        # LeakyRelu acc scales for dense layers
        for name, xs, ys in dense_leakyrelu_scales:
            acc = xs / ys
            sname = name.upper().replace(".", "_") + "_LEAKYRELU_QUANT_ACC_SCALE"
            f.write(f"// /{name}/LeakyRelu_quant (QLinearLeakyRelu)\n")
            f.write(f"// x_scale={xs}, w_scale=None, y_scale={ys}\n")
            f.write(f"#define {sname} {acc:.10e}f\n\n")

        f.write("\n")

        # --- Conv2 acc scale ---
        conv2_acc = conv2_x_scale * conv2_w_scale / conv2_y_scale
        f.write(f"// /cnn_layers.2/Conv_quant (QLinearConv)\n")
        f.write(f"// x_scale={conv2_x_scale}, w_scale={conv2_w_scale}, y_scale={conv2_y_scale}\n")
        f.write(f"#define CNN_LAYERS_2_CONV_QUANT_ACC_SCALE {conv2_acc:.10e}f\n\n")

        # --- Conv3 acc scale ---
        conv3_acc = conv3_x_scale * conv3_w_scale / conv3_y_scale
        f.write(f"// /cnn_layers.4/Conv_quant (QLinearConv)\n")
        f.write(f"// x_scale={conv3_x_scale}, w_scale={conv3_w_scale}, y_scale={conv3_y_scale}\n")
        f.write(f"#define CNN_LAYERS_4_CONV_QUANT_ACC_SCALE {conv3_acc:.10e}f\n\n")

        # --- FC layer acc scales and individual scales ---
        f.write("\n// ============================================\n")
        f.write("// FC (Gemm) Layers\n")
        f.write("// Weight scales calculated from float weights\n")
        f.write("// ============================================\n\n")

        fc_c_names = [
            ("DENSE_LAYERS_0_GEMM", "fc1", 16, 200),
            ("DENSE_LAYERS_2_GEMM", "fc2", 8, 16),
            ("DENSE_LAYERS_4_GEMM", "fc3", 4, 8),
            ("DENSE_LAYERS_6_GEMM", "fc4", 2, 4),
            ("DENSE_LAYERS_8_GEMM", "output", 2, 2),
        ]

        for (prefix, _, _, _), fcr in zip(fc_c_names, fc_results):
            f.write(f"// /{fcr['name']}/Gemm (Gemm (FC)) - weight shape: {list(fcr['int8_w'].shape)}\n")
            f.write(f"// x_scale={fcr['x_scale']:.10e} w_scale={fcr['w_scale']:.10e} y_scale={fcr['y_scale']:.10e}\n")
            f.write(f"#define {prefix}_ACC_SCALE {fcr['acc_scale']:.10e}f\n\n")

        f.write("\n// ============================================\n")
        f.write("// FC Layer Individual Scales\n")
        f.write("// ============================================\n\n")

        for (prefix, _, _, _), fcr in zip(fc_c_names, fc_results):
            f.write(f"// /{fcr['name']}/Gemm\n")
            f.write(f"#define {prefix}_X_SCALE {fcr['x_scale']:.10e}f\n")
            f.write(f"#define {prefix}_W_SCALE {fcr['w_scale']:.10e}f\n")
            f.write(f"#define {prefix}_Y_SCALE {fcr['y_scale']:.10e}f\n\n")

        # =====================================================================
        # 5. Write weight arrays
        # =====================================================================
        f.write("\n//\n//\n// INT8\n//\n//\n\n")

        # Conv1 weights
        f.write(f"static elem_t\n")
        f.write(f"    conv1_weights[CONV1_FILTERS][CONV1_KERNEL][CONV1_KERNEL][INPUT_CHANNELS] = {{\n")
        f.write(format_conv_weights_okkc(conv1_w))
        f.write(f"\n     }};\n\n")

        # Conv1 bias
        f.write(f"static acc_t conv1_bias[CONV1_FILTERS] =\n")
        f.write(f"  {{\n")
        f.write(format_1d_array(conv1_b))
        f.write(f"\n  }};\n\n")

        # Conv2 weights
        f.write(f"static elem_t conv2_weights[CONV2_FILTERS][CONV2_KERNEL][CONV2_KERNEL]\n")
        f.write(f"                           [CONV1_FILTERS] = \n{{\n")
        f.write(format_conv_weights_okkc(conv2_w))
        f.write(f"\n}};\n\n")

        # Conv2 bias
        f.write(f"static acc_t conv2_bias[CONV2_FILTERS] =\n")
        f.write(f"  {{\n")
        f.write(format_1d_array(conv2_b))
        f.write(f"\n  }};\n\n")

        # Conv3 weights
        f.write(f"static elem_t conv3_weights[CONV3_FILTERS][CONV3_KERNEL][CONV3_KERNEL]\n")
        f.write(f"                           [CONV2_FILTERS] = \n{{\n")
        f.write(format_conv_weights_okkc(conv3_w))
        f.write(f"\n}};\n\n")

        # Conv3 bias
        f.write(f"static acc_t conv3_bias[CONV3_FILTERS] =\n")
        f.write(f"  {{\n")
        f.write(format_1d_array(conv3_b))
        f.write(f"\n  }};\n\n")

        # NLB theta weights [32][64]
        f.write(f"static elem_t nlb_theta_weights[CONV2_FILTERS][CONV1_FILTERS] = {{\n")
        f.write(format_1x1_conv_weights(nlb_theta_w))
        f.write(f"\n}};\n\n")

        # NLB theta bias
        f.write(f"static acc_t nlb_theta_bias[CONV2_FILTERS] =\n")
        f.write(f"     {{\n")
        f.write(format_1d_array(nlb_theta_b, indent="         "))
        f.write(f"\n     }};\n\n")

        # NLB phi weights [32][64]
        f.write(f"static elem_t nlb_phi_weights[CONV2_FILTERS][CONV1_FILTERS] = {{\n")
        f.write(format_1x1_conv_weights(nlb_phi_w))
        f.write(f"\n}};\n\n")

        # NLB phi bias
        f.write(f"static acc_t nlb_phi_bias[CONV2_FILTERS] = {{\n")
        f.write(format_1d_array(nlb_phi_b, indent="         "))
        f.write(f"\n}};\n\n")

        # NLB g weights [32][64]
        f.write(f"static elem_t nlb_g_weights[CONV2_FILTERS][CONV1_FILTERS] = {{\n")
        f.write(format_1x1_conv_weights(nlb_g_w))
        f.write(f"\n}};\n\n")

        # NLB g bias
        f.write(f"static acc_t nlb_g_bias[CONV2_FILTERS] = {{\n")
        f.write(format_1d_array(nlb_g_b, indent="       "))
        f.write(f"\n}};\n\n")

        # NLB out weights [64][32]
        f.write(f"static elem_t nlb_out_weights[CONV1_FILTERS][CONV2_FILTERS] = {{\n")
        f.write(format_1x1_conv_weights(nlb_out_w))
        f.write(f"\n}};\n\n")

        # NLB out bias
        f.write(f"static acc_t nlb_out_bias[CONV1_FILTERS] = {{\n")
        f.write(format_1d_array(nlb_out_b, indent="       "))
        f.write(f"\n}};\n\n")

        # FC1 weights [16][200]
        f.write(f"static elem_t fc1_weights[FC1_UNITS][CONV3_FLATTENED] = {{\n")
        f.write(format_fc_weights(fc_results[0]["int8_w"]))
        f.write(f"\n}};\n\n")

        # FC1 bias
        f.write(f"static acc_t fc1_bias[FC1_UNITS] = {{\n")
        f.write(format_1d_array(fc_results[0]["int32_b"], indent="     "))
        f.write(f"\n}};\n\n")

        # FC2 weights [8][16]
        f.write(f"static elem_t fc2_weights[FC2_UNITS][FC1_UNITS] = {{\n")
        f.write(format_fc_weights(fc_results[1]["int8_w"]))
        f.write(f"\n}};\n\n")

        # FC2 bias
        f.write(f"static acc_t fc2_bias[FC2_UNITS] = {{\n")
        f.write(format_1d_array(fc_results[1]["int32_b"], indent="       "))
        f.write(f"\n}};\n\n")

        # FC3 weights [4][8]
        f.write(f"static elem_t fc3_weights[FC3_UNITS][FC2_UNITS] = {{\n")
        f.write(format_fc_weights(fc_results[2]["int8_w"]))
        f.write(f"\n}};\n\n")

        # FC3 bias
        f.write(f"static acc_t fc3_bias[FC3_UNITS] = {{\n")
        f.write(format_1d_array(fc_results[2]["int32_b"], indent="     "))
        f.write(f"\n}};\n\n")

        # FC4 weights [2][4]
        f.write(f"static elem_t fc4_weights[FC4_UNITS][FC3_UNITS] = {{\n")
        f.write(format_fc_weights(fc_results[3]["int8_w"]))
        f.write(f"\n}};\n\n")

        # FC4 bias
        f.write(f"static acc_t fc4_bias[FC4_UNITS] = {{\n")
        f.write(format_1d_array(fc_results[3]["int32_b"], indent="      "))
        f.write(f"\n}};\n\n")

        # Output weights [2][2]
        f.write(f"static elem_t output_weights[OUTPUT_UNITS][FC4_UNITS] = {{\n")
        f.write(format_fc_weights(fc_results[4]["int8_w"]))
        f.write(f"\n}};\n\n")

        # Output bias
        f.write(f"static acc_t output_bias[OUTPUT_UNITS]= {{\n")
        f.write(format_1d_array(fc_results[4]["int32_b"], indent="      "))
        f.write(f"\n}};\n")

    print(f"Done! Header written to {output_path}")

    # Print verification summary
    print("\n=== Verification Summary ===")
    print(f"Conv1 weights: {conv1_w.shape} -> [{conv1_w.shape[0]}][{conv1_w.shape[2]}][{conv1_w.shape[3]}][{conv1_w.shape[1]}]")
    print(f"Conv2 weights: {conv2_w.shape} -> [{conv2_w.shape[0]}][{conv2_w.shape[2]}][{conv2_w.shape[3]}][{conv2_w.shape[1]}]")
    print(f"Conv3 weights: {conv3_w.shape} -> [{conv3_w.shape[0]}][{conv3_w.shape[2]}][{conv3_w.shape[3]}][{conv3_w.shape[1]}]")
    print(f"NLB theta: {nlb_theta_w.shape} -> [{nlb_theta_w.shape[0]}][{nlb_theta_w.shape[1]}]")
    print(f"NLB phi:   {nlb_phi_w.shape} -> [{nlb_phi_w.shape[0]}][{nlb_phi_w.shape[1]}]")
    print(f"NLB g:     {nlb_g_w.shape} -> [{nlb_g_w.shape[0]}][{nlb_g_w.shape[1]}]")
    print(f"NLB out:   {nlb_out_w.shape} -> [{nlb_out_w.shape[0]}][{nlb_out_w.shape[1]}]")
    for (prefix, cname, o, i), fcr in zip(fc_c_names, fc_results):
        print(f"FC {cname}: {fcr['int8_w'].shape} (w_scale={fcr['w_scale']:.6e})")


if __name__ == "__main__":
    main()
