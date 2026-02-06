#!/usr/bin/env python3
"""
Extract quantization scales from ONNX model and calculate acc_scale for each layer.

acc_scale = (x_scale * w_scale) / y_scale

This is the requantization scale needed to convert Int32 accumulator back to Int8.
"""

import onnx
import numpy as np
from onnx import numpy_helper


def extract_scales(model_path: str) -> dict[str, float]:
    """Extract all scale values from ONNX model initializers."""
    model = onnx.load(model_path)
    scales = {}
    for init in model.graph.initializer:
        if "scale" in init.name:
            arr = numpy_helper.to_array(init)
            if arr.ndim == 0:
                scales[init.name] = float(arr)
            else:
                scales[init.name] = arr.tolist()
    return scales


def extract_weights(model_path: str) -> dict[str, np.ndarray]:
    """Extract all weight tensors from ONNX model initializers."""
    model = onnx.load(model_path)
    weights = {}
    for init in model.graph.initializer:
        if "weight" in init.name and "scale" not in init.name and "zero_point" not in init.name:
            weights[init.name] = numpy_helper.to_array(init)
    return weights


def calculate_weight_scale(weights: np.ndarray) -> float:
    """Calculate symmetric quantization scale for weights.

    scale = max(abs(weights)) / 127
    """
    max_val = np.max(np.abs(weights))
    if max_val == 0:
        return 1.0
    return float(max_val / 127.0)


def get_qlinear_ops(model_path: str) -> list[dict]:
    """Extract QLinear operations and their input/output scale names."""
    model = onnx.load(model_path)
    ops = []

    for node in model.graph.node:
        if node.op_type == "QLinearConv":
            ops.append({
                "name": node.name,
                "op_type": node.op_type,
                "x_scale": node.input[1],
                "w_scale": node.input[4],
                "y_scale": node.input[6],
            })
        elif node.op_type == "QLinearMatMul":
            ops.append({
                "name": node.name,
                "op_type": node.op_type,
                "x_scale": node.input[1],
                "w_scale": node.input[4],
                "y_scale": node.input[6],
            })
        elif node.op_type == "QLinearAdd":
            ops.append({
                "name": node.name,
                "op_type": node.op_type,
                "x_scale": node.input[1],
                "w_scale": node.input[4],
                "y_scale": node.input[6],
            })
        elif node.op_type == "QLinearLeakyRelu":
            ops.append({
                "name": node.name,
                "op_type": node.op_type,
                "x_scale": node.input[1],
                "w_scale": None,
                "y_scale": node.input[3],
            })

    return ops


def get_gemm_ops(model_path: str) -> list[dict]:
    """Extract Gemm (FC) operations."""
    model = onnx.load(model_path)
    ops = []

    for node in model.graph.node:
        if node.op_type == "Gemm":
            ops.append({
                "name": node.name,
                "op_type": node.op_type,
                "input": node.input[0],
                "weight": node.input[1],
                "bias": node.input[2] if len(node.input) > 2 else None,
                "output": node.output[0],
            })

    return ops


def calculate_acc_scales(model_path: str) -> list[dict]:
    """Calculate acc_scale for each QLinear operation."""
    scales = extract_scales(model_path)
    ops = get_qlinear_ops(model_path)

    results = []
    for op in ops:
        x_scale = scales.get(op["x_scale"])
        w_scale = scales.get(op["w_scale"]) if op["w_scale"] else None
        y_scale = scales.get(op["y_scale"])

        if x_scale is not None and y_scale is not None:
            if w_scale is not None:
                if isinstance(w_scale, list):
                    acc_scale = [(x_scale * ws) / y_scale for ws in w_scale]
                else:
                    acc_scale = (x_scale * w_scale) / y_scale
            else:
                acc_scale = x_scale / y_scale

            results.append({
                "name": op["name"],
                "op_type": op["op_type"],
                "x_scale": x_scale,
                "w_scale": w_scale,
                "y_scale": y_scale,
                "acc_scale": acc_scale,
            })

    return results


def calculate_fc_scales(model_path: str) -> list[dict]:
    """Calculate acc_scale for FC (Gemm) layers.

    FC layers in this model are not quantized in ONNX, so we need to:
    1. Calculate weight scale from float weights
    2. Get input scale from previous layer's output scale
    3. Get output scale from QuantizeLinear after Gemm
    """
    scales = extract_scales(model_path)
    weights = extract_weights(model_path)
    gemm_ops = get_gemm_ops(model_path)

    # Map input names to their scales
    # FC input comes from DequantizeLinear output, which uses the previous layer's scale
    input_scale_map = {
        # FC1: input from Flatten (after cnn_layers.5/LeakyRelu)
        "/Flatten_output_0": "/cnn_layers.5/LeakyRelu_output_0_scale",
        # FC2: input from dense_layers.1/LeakyRelu
        "/dense_layers.1/LeakyRelu_output_0": "/dense_layers.1/LeakyRelu_output_0_scale",
        # FC3: input from dense_layers.3/LeakyRelu
        "/dense_layers.3/LeakyRelu_output_0": "/dense_layers.3/LeakyRelu_output_0_scale",
        # FC4: input from dense_layers.5/LeakyRelu
        "/dense_layers.5/LeakyRelu_output_0": "/dense_layers.5/LeakyRelu_output_0_scale",
        # FC5 (output): input from dense_layers.7/LeakyRelu
        "/dense_layers.7/LeakyRelu_output_0": "/dense_layers.7/LeakyRelu_output_0_scale",
    }

    # Map Gemm output to its QuantizeLinear scale
    output_scale_map = {
        "/dense_layers.0/Gemm_output_0": "/dense_layers.0/Gemm_output_0_scale",
        "/dense_layers.2/Gemm_output_0": "/dense_layers.2/Gemm_output_0_scale",
        "/dense_layers.4/Gemm_output_0": "/dense_layers.4/Gemm_output_0_scale",
        "/dense_layers.6/Gemm_output_0": "/dense_layers.6/Gemm_output_0_scale",
        "output": None,  # Final output layer - no quantization after
    }

    results = []
    for op in gemm_ops:
        weight_name = op["weight"]
        input_name = op["input"]
        output_name = op["output"]

        # Get weight and calculate weight scale
        w = weights.get(weight_name)
        if w is None:
            continue
        w_scale = calculate_weight_scale(w)

        # Get input scale
        x_scale_name = input_scale_map.get(input_name)
        x_scale = scales.get(x_scale_name) if x_scale_name else None

        # Get output scale
        y_scale_name = output_scale_map.get(output_name)
        y_scale = scales.get(y_scale_name) if y_scale_name else None

        # For the final output layer, use y_scale = 1/127 so that
        # int8 output can represent [0, 1] range (int8=127 -> float=1.0)
        if y_scale is None:
            y_scale = 1.0 / 127.0  # int8=127 corresponds to float=1.0

        if x_scale is not None:
            acc_scale = (x_scale * w_scale) / y_scale

            results.append({
                "name": op["name"],
                "op_type": "Gemm (FC)",
                "x_scale": x_scale,
                "x_scale_name": x_scale_name,
                "w_scale": w_scale,
                "w_scale_calculated": True,
                "y_scale": y_scale,
                "y_scale_name": y_scale_name,
                "acc_scale": acc_scale,
                "weight_shape": w.shape,
            })

    return results


def generate_c_header(qlinear_results: list[dict], fc_results: list[dict]) -> str:
    """Generate C header definitions for acc_scales."""
    lines = [
        "// Auto-generated quantization scales",
        "// acc_scale = (x_scale * w_scale) / y_scale",
        "//",
        "// This scale converts Int32 accumulator to Int8 output",
        "",
        "// ============================================",
        "// QLinear Operations (Conv, MatMul, Add, etc.)",
        "// ============================================",
        "",
    ]

    for r in qlinear_results:
        name = r["name"]
        c_name = name.replace("/", "_").replace(".", "_").strip("_")
        c_name = c_name.upper() + "_ACC_SCALE"

        acc_scale = r["acc_scale"]
        if isinstance(acc_scale, list):
            lines.append(f"// {name} ({r['op_type']})")
            lines.append(f"// x_scale={r['x_scale']}, y_scale={r['y_scale']}")
            lines.append(f"// w_scale (per-channel): {len(acc_scale)} values")
            vals = ", ".join(f"{v:.10e}f" for v in acc_scale)
            lines.append(f"static const float {c_name}[] = {{{vals}}};")
        else:
            lines.append(f"// {name} ({r['op_type']})")
            lines.append(f"// x_scale={r['x_scale']}, w_scale={r['w_scale']}, y_scale={r['y_scale']}")
            lines.append(f"#define {c_name} {acc_scale:.10e}f")
        lines.append("")

    lines.extend([
        "",
        "// ============================================",
        "// FC (Gemm) Layers",
        "// Weight scales calculated from float weights",
        "// ============================================",
        "",
    ])

    for r in fc_results:
        name = r["name"]
        c_name = name.replace("/", "_").replace(".", "_").strip("_")
        c_name = c_name.upper() + "_ACC_SCALE"

        lines.append(f"// {name} ({r['op_type']}) - weight shape: {r['weight_shape']}")
        lines.append(f"// x_scale={r['x_scale']:.10e} (from {r['x_scale_name']})")
        lines.append(f"// w_scale={r['w_scale']:.10e} (calculated from weights)")
        if r['y_scale_name']:
            lines.append(f"// y_scale={r['y_scale']:.10e} (from {r['y_scale_name']})")
        else:
            lines.append(f"// y_scale={r['y_scale']:.10e} (output layer - no requantization)")
        lines.append(f"#define {c_name} {r['acc_scale']:.10e}f")
        lines.append("")

    # Also output individual scales for FC layers (useful for weight quantization)
    lines.extend([
        "",
        "// ============================================",
        "// FC Layer Individual Scales",
        "// ============================================",
        "",
    ])

    for r in fc_results:
        name = r["name"]
        base_name = name.replace("/", "_").replace(".", "_").strip("_").upper()
        lines.append(f"// {name}")
        lines.append(f"#define {base_name}_X_SCALE {r['x_scale']:.10e}f")
        lines.append(f"#define {base_name}_W_SCALE {r['w_scale']:.10e}f")
        lines.append(f"#define {base_name}_Y_SCALE {r['y_scale']:.10e}f")
        lines.append("")

    return "\n".join(lines)


def main():
    model_path = "onnx/braggnn_int8_qlinear_opset10.onnx"

    print("=" * 60)
    print("Extracting scales from:", model_path)
    print("=" * 60)

    # Extract and display all scales
    print("\n[All Scales in Model]")
    print("-" * 40)
    scales = extract_scales(model_path)
    for name, value in sorted(scales.items()):
        if isinstance(value, list):
            print(f"{name}: [{len(value)} values]")
        else:
            print(f"{name}: {value}")

    # Calculate acc_scales for QLinear ops
    print("\n" + "=" * 60)
    print("[QLinear Operations - acc_scale]")
    print("acc_scale = (x_scale * w_scale) / y_scale")
    print("-" * 40)

    qlinear_results = calculate_acc_scales(model_path)
    for r in qlinear_results:
        print(f"\n{r['name']} ({r['op_type']})")
        print(f"  x_scale: {r['x_scale']}")
        if r['w_scale'] is not None:
            if isinstance(r['w_scale'], list):
                print(f"  w_scale: [{len(r['w_scale'])} per-channel values]")
            else:
                print(f"  w_scale: {r['w_scale']}")
        print(f"  y_scale: {r['y_scale']}")
        if isinstance(r['acc_scale'], list):
            print(f"  acc_scale: [{len(r['acc_scale'])} per-channel values]")
        else:
            print(f"  acc_scale: {r['acc_scale']}")

    # Calculate acc_scales for FC (Gemm) layers
    print("\n" + "=" * 60)
    print("[FC (Gemm) Layers - acc_scale]")
    print("Weight scales calculated from float32 weights")
    print("-" * 40)

    fc_results = calculate_fc_scales(model_path)
    for r in fc_results:
        print(f"\n{r['name']} ({r['op_type']}) - weight shape: {r['weight_shape']}")
        print(f"  x_scale: {r['x_scale']:.10e} (from {r['x_scale_name']})")
        print(f"  w_scale: {r['w_scale']:.10e} (calculated)")
        if r['y_scale_name']:
            print(f"  y_scale: {r['y_scale']:.10e} (from {r['y_scale_name']})")
        else:
            print(f"  y_scale: {r['y_scale']:.10e} (output layer)")
        print(f"  acc_scale: {r['acc_scale']:.10e}")

    # Generate C header
    print("\n" + "=" * 60)
    print("[C Header Definitions]")
    print("-" * 40)
    c_header = generate_c_header(qlinear_results, fc_results)
    print(c_header)

    # Save to file
    header_path = "braggnn_scales.h"
    with open(header_path, "w") as f:
        f.write(c_header)
    print(f"\nSaved to: {header_path}")

    # Also print quantized weights info for FC layers
    print("\n" + "=" * 60)
    print("[FC Weight Quantization Info]")
    print("To quantize float weights to int8:")
    print("  int8_weight = round(float_weight / w_scale)")
    print("-" * 40)
    weights = extract_weights(model_path)
    for r in fc_results:
        weight_name = r['name'].replace('/Gemm', '').replace('/', '.').strip('.') + ".weight"
        # Fix the weight name format
        weight_name = weight_name.replace("dense_layers", "dense_layers")
        w = None
        for wn, wv in weights.items():
            if "dense" in wn and wn == weight_name.replace("/", ".").lstrip("."):
                w = wv
                break
        if w is None:
            # Try another format
            for wn, wv in weights.items():
                if r['weight_shape'] == wv.shape and "dense" in wn:
                    w = wv
                    weight_name = wn
                    break

        if w is not None:
            w_int8 = np.clip(np.round(w / r['w_scale']), -128, 127).astype(np.int8)
            print(f"\n{weight_name}:")
            print(f"  Original shape: {w.shape}")
            print(f"  w_scale: {r['w_scale']:.10e}")
            print(f"  int8 range: [{w_int8.min()}, {w_int8.max()}]")


if __name__ == "__main__":
    main()
