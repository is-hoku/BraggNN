import onnx
from onnx import version_converter

def main():
    m = onnx.load("onnx/braggnn_fp32.onnx")
    converted_model = version_converter.convert_version(m, 12)
    onnx.save(converted_model, "onnx/braggnn_fp32_12.onnx")

    m = onnx.load("onnx/braggnn_int8.onnx")
    converted_model = version_converter.convert_version(m, 12)
    onnx.save(converted_model, "onnx/braggnn_int8_12.onnx")

if __name__ == "__main__":
    main()
