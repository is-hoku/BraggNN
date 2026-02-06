#!/usr/bin/env python3
"""Extract all intermediate values from BraggNN ONNX model for comparison with Gemmini."""

import numpy as np
import onnx
from onnx import helper, TensorProto, shape_inference
import onnxruntime as ort

# Load model and fix opset
model_path = "/home/hoku/repo/BraggNN/onnx/braggnn_int8_qlinear_opset10.onnx"
model = onnx.load(model_path)
model.opset_import[0].version = 12

# Run shape inference to get proper types for all intermediate tensors
model = shape_inference.infer_shapes(model)

# Collect all intermediate tensor names and their value_info
vi_map = {}
for vi in model.graph.value_info:
    vi_map[vi.name] = vi

# Add all intermediate node outputs to the graph outputs
existing_output_names = {o.name for o in model.graph.output}
for node in model.graph.node:
    for output in node.output:
        if output and output not in existing_output_names and output in vi_map:
            model.graph.output.append(vi_map[output])
            existing_output_names.add(output)

# Save modified model
tmp_path = "/tmp/claude-1000/-home-hoku-repo-chipyard-generators-gemmini-software-gemmini-rocc-tests-bareMetalC/481e6352-f6d7-47dd-b984-a2b9f5e24e35/scratchpad/braggnn_debug.onnx"
onnx.save(model, tmp_path)

# Create session
sess = ort.InferenceSession(tmp_path)

# Input data (same as braggnn.c)
fp32_input = np.array([
    [0.00000000, 0.01542046, 0.03264448, 0.05223434, 0.07307496, 0.09205172, 0.10619149, 0.01651590, 0.02395481, 0.03264448, 0.04369999],
    [0.05783976, 0.07644960, 0.10135448, 0.03553860, 0.07306661, 0.10408917, 0.11822893, 0.11548591, 0.10623743, 0.10135448, 0.00575078],
    [0.01850472, 0.04479543, 0.08819781, 0.15404664, 0.23176228, 0.29179421, 0.20695563, 0.17520320, 0.12576711, 0.08819781, 0.07307496],
    [0.08051388, 0.12037718, 0.09748757, 0.22109538, 0.40085012, 0.53430861, 0.51267964, 0.37779671, 0.21994427, 0.13236861, 0.09748757],
    [0.09744163, 0.09600913, 0.20195234, 0.42009991, 0.70064259, 0.89028502, 0.69816506, 0.51993632, 0.30917105, 0.19282073, 0.06591162],
    [0.10261316, 0.12576711, 0.25217456, 0.49831432, 0.82340622, 1.00000000, 0.79662609, 0.57843238, 0.33595118, 0.19282073, 0.13236861],
    [0.08050554, 0.10619149, 0.20695563, 0.41752210, 0.65042037, 0.79662609, 0.72853380, 0.49831432, 0.30917105, 0.21994427, 0.13236861],
    [0.06591162, 0.09748757, 0.17520320, 0.30917105, 0.49831432, 0.57843238, 0.49831432, 0.37779671, 0.21994427, 0.09748757, 0.06591162],
    [0.03264448, 0.06591162, 0.12576711, 0.21994427, 0.30917105, 0.33595118, 0.30917105, 0.21994427, 0.12576711, 0.06591162, 0.03264448],
    [0.00000000, 0.03264448, 0.06591162, 0.09748757, 0.19282073, 0.19282073, 0.19282073, 0.09748757, 0.06591162, 0.03264448, 0.00000000],
    [0.00575078, 0.01542046, 0.03264448, 0.06591162, 0.06591162, 0.13236861, 0.06591162, 0.06591162, 0.03264448, 0.01542046, 0.00575078],
], dtype=np.float32).reshape(1, 1, 11, 11)

# Run inference
input_name = sess.get_inputs()[0].name
output_names = [o.name for o in sess.get_outputs()]
results = sess.run(output_names, {input_name: fp32_input})

# Build name->value mapping
values = {}
for name, val in zip(output_names, results):
    values[name] = val

# Print node listing first
print("=" * 80)
print("NODE LISTING:")
print("=" * 80)
for i, node in enumerate(model.graph.node):
    outs = ", ".join(node.output)
    ins_short = ", ".join(node.input[:4])
    if len(node.input) > 4:
        ins_short += ", ..."
    print(f"[{i:3d}] {node.op_type:25s} ({ins_short}) -> {outs}")

# Print key intermediate values
print("\n" + "=" * 80)
print("INTERMEDIATE VALUES IN EXECUTION ORDER:")
print("=" * 80)

for i, node in enumerate(model.graph.node):
    for out in node.output:
        if out not in values:
            continue
        v = values[out]
        if v.size == 0:
            continue

        label = f"[{i:3d}] {node.op_type:20s} {out}"

        if v.ndim == 4:
            n, c, h, w = v.shape
            print(f"\n{label}  shape={v.shape} dtype={v.dtype}")
            nc = min(8, c)
            # NCHW: ch0-7@(0,0) means v[0, :8, 0, 0]
            print(f"      ch0-{nc-1}@(0,0): {v[0,:nc,0,0].tolist()}")
            if h > 4 and w > 4:
                print(f"      ch0-{nc-1}@(4,4): {v[0,:nc,4,4].tolist()}")
            if h > 1:
                print(f"      ch0@(1,0)={v[0,0,1,0]}")
            if h > 4 and w > 4:
                print(f"      ch0@(4,4)={v[0,0,4,4]}")
            if h > 8 and w > 8:
                print(f"      ch0@(8,8)={v[0,0,8,8]}")
        elif v.ndim == 3:
            print(f"\n{label}  shape={v.shape} dtype={v.dtype}")
            flat = v.flatten()
            print(f"      first 16: {flat[:min(16,len(flat))].tolist()}")
        elif v.ndim == 2:
            print(f"\n{label}  shape={v.shape} dtype={v.dtype}")
            flat = v.flatten()
            if len(flat) <= 32:
                print(f"      ALL: {flat.tolist()}")
            else:
                print(f"      first 16: {flat[:16].tolist()}")
                print(f"      row0[:3]: {v[0,:min(3,v.shape[1])].tolist()}")
        elif v.ndim == 1:
            print(f"\n{label}  shape={v.shape} dtype={v.dtype}")
            if v.size <= 32:
                print(f"      ALL: {v.tolist()}")
            else:
                print(f"      first 16: {v[:16].tolist()}")
        elif v.ndim == 0 or v.size == 1:
            print(f"\n{label}  value={v.item()}")

# Print final output
print("\n\n" + "=" * 80)
print("FINAL OUTPUT:")
print("=" * 80)
original_output_names = ["output"]  # typical name
for name in output_names[:3]:  # check first few
    if name in values:
        v = values[name]
        if v.ndim <= 2 and v.size <= 10:
            print(f"  {name}: {v.flatten().tolist()} shape={v.shape} dtype={v.dtype}")

# Dequantize final output if needed
# Look for the last node's output
last_nodes = list(model.graph.node)[-5:]
print("\nLast 5 nodes outputs:")
for node in last_nodes:
    for out in node.output:
        if out in values:
            v = values[out]
            print(f"  {node.op_type} -> {out}: {v.flatten()[:10].tolist()} dtype={v.dtype}")
