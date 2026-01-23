import numpy as np
import onnx
from onnx import shape_inference
import torch
from model import BraggNN
from onnxruntime.quantization import (
    quantize_static,
    QuantFormat,
    QuantType,
    CalibrationDataReader,
)

def make_gaussian(imgsz=11, x_cen=6.0, y_cen=5.0, sig_x=0.6, sig_y=1.5, amp=1000.0, norm=True):
    x = np.arange(imgsz); y = np.arange(imgsz)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = np.exp(-((X-x_cen)**2)/(2*sig_x**2) - ((Y-y_cen)**2)/(2*sig_y**2))
    Z *= amp
    if norm:
        zmin, zmax = Z.min(), Z.max()
        if zmax > zmin: Z = (Z - zmin) / (zmax - zmin)
    return Z.astype(np.float32)


class BraggNNCalibrationDataReader(CalibrationDataReader):
    def __init__(self, imgsz=11, num_samples=128, seed=42):
        self.imgsz = imgsz
        self.num_samples = num_samples
        self.rng = np.random.default_rng(seed)
        self.current_idx = 0

        self.data = []
        for _ in range(num_samples):
            x_cen = float(self.rng.uniform(0.5, imgsz - 1.5))
            y_cen = float(self.rng.uniform(0.5, imgsz - 1.5))
            sig_x = float(self.rng.uniform(0.3, 1.0))
            sig_y = float(self.rng.uniform(0.5, 2.0))
            img = make_gaussian(imgsz, x_cen, y_cen, sig_x, sig_y)
            self.data.append(img[None, None])

    def get_next(self):
        if self.current_idx >= self.num_samples:
            return None
        data = self.data[self.current_idx]
        self.current_idx += 1
        return {'input': data}

    def rewind(self):
        self.current_idx = 0


def main():
    imgsz = 11
    fp32_onnx_path = "onnx/braggnn_fp32_opset10.onnx"
    int8_onnx_path = "onnx/braggnn_int8_qdq_opset10.onnx"

    # Load FP32 model
    print("Loading FP32 PyTorch model...")
    model = BraggNN(imgsz=imgsz, fcsz=(16, 8, 4, 2)).eval()
    model.load_state_dict(torch.load('models/fc16_8_4_2-sz11.pth', map_location='cpu', weights_only=True))

    # Example input
    example_input = torch.from_numpy(make_gaussian(imgsz)[None, None])

    # Export FP32 to ONNX opset 10
    print(f"Exporting FP32 model to {fp32_onnx_path}...")
    torch.onnx.export(
        model,
        example_input,
        fp32_onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=10,
        export_params=True,
        do_constant_folding=True,
        dynamo=False,
    )
    print("FP32 export done.")

    # Run shape inference on the FP32 model
    #print("Running shape inference...")
    #model_onnx = onnx.load(fp32_onnx_path)
    #model_onnx = shape_inference.infer_shapes(model_onnx)
    #onnx.save(model_onnx, fp32_onnx_path)
    #print("Shape inference done.")

    # Static quantization with QLinearOps format
    print(f"Quantizing to {int8_onnx_path} with QLinearOps format...")
    calibration_reader = BraggNNCalibrationDataReader(imgsz=imgsz, num_samples=128)

    quantize_static(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        calibration_data_reader=calibration_reader,
        #quant_format=QuantFormat.QOperator,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        #op_types_to_quantize=['Conv', 'MatMul', 'Gemm', 'Gather', 'Unsqueeze', 'Concat', 'Reshape', 'Transpose', 'Softmax', 'LeakyRelu', 'Add', 'Flatten'],
        op_types_to_quantize=['Conv', 'MatMul', 'Gather', 'Unsqueeze', 'Concat', 'Reshape', 'Transpose', 'LeakyRelu', 'Add', 'Flatten'],
        extra_options={
            'ActivationSymmetric': True,
            'WeightSymmetric': True,
        },
    )
    print("INT8 static quantization done.")

    # Run shape inference on quantized model to populate value_info with INT8 types
    #print("Running shape inference on quantized model...")
    #model_onnx = onnx.load(int8_onnx_path)
    #model_onnx = shape_inference.infer_shapes(model_onnx)
    #onnx.save(model_onnx, int8_onnx_path)
    #print("Shape inference done.")

    # Verify the model structure
    model_onnx = onnx.load(int8_onnx_path)
    onnx.checker.check_model(model_onnx)
    print(f"\nModel valid: Yes")
    print(f"Opset version: {model_onnx.opset_import[0].version}")

    # Count operators
    from collections import Counter
    op_types = [n.op_type for n in model_onnx.graph.node]
    counts = Counter(op_types)
    print(f"\nOperator counts:")
    for op, count in sorted(counts.items()):
        print(f"  {op}: {count}")

    # Check for QLinear ops
    qlinear_ops = [op for op in counts.keys() if 'QLinear' in op]
    if qlinear_ops:
        print(f"\nQLinear operators found (onnxruntime-riscv compatible):")
        for op in qlinear_ops:
            print(f"  {op}: {counts[op]}")
    else:
        print("\nWarning: No QLinear operators found!")

if __name__ == "__main__":
    main()
