import torch
import torchao.quantization.pt2e
from model import BraggNN

def main():
    example_inputs = (torch.randn(1, 1, 11, 11),)

    # fp32
    fp32_model = BraggNN(imgsz=11, fcsz=(16, 8, 4, 2)).eval()
    fp32_model.load_state_dict(torch.load('models/fc16_8_4_2-sz11.pth', map_location=torch.device('cpu')))
    fp32_output_path = "onnx/braggnn_fp32.onnx"

    torch.onnx.export(
        fp32_model,
        example_inputs[0],
        fp32_output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=13, # For use old opsets
        export_params=True,
        do_constant_folding=True,
        #dynamo=False, # For use old APIs
    )

    # int8
    int8_module = torch.export.load('models/int8_16_8_4_2-sz11.pth').module()
    #int8_module = torch.load(
    #    'models/int8_16_8_4_2-sz11.pth',
    #    map_location=torch.device('cpu'),
    #).eval()
    int8_output_path = "onnx/braggnn_int8.onnx"

    torch.onnx.export(
        int8_module,
        example_inputs[0],
        int8_output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=13, # For use old opsets
        export_params=True,
        do_constant_folding=True,
        #dynamo=False, # For use old APIs
    )

if __name__ == "__main__":
    main()