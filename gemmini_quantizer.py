import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'executorch'))
import numpy as np
import torch
from torch.export import export
from model import BraggNN
from torch.utils.data import DataLoader
from dataset import BraggNNDataset

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
import tvm.relax.backend.contrib.gemmini


def make_gaussian(imgsz=11, x_cen=6.0, y_cen=5.0, sig_x=0.6, sig_y=1.5, amp=1000.0, norm=True):
    x = np.arange(imgsz); y = np.arange(imgsz)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = np.exp(-((X-x_cen)**2)/(2*sig_x**2) - ((Y-y_cen)**2)/(2*sig_y**2))
    Z *= amp
    if norm:
        zmin, zmax = Z.min(), Z.max()
        if zmax > zmin: Z = (Z - zmin) / (zmax - zmin)
    return Z.astype(np.float32)

def calibrate(model, data_loader):
    from torchao.quantization.pt2e import allow_exported_model_train_eval
    allow_exported_model_train_eval(model)
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            for i in range(image.shape[0]):
                model(image[i:i+1])

def main():
    ds_input = BraggNNDataset(psz=11, rnd_shift=0, use='validation')
    dl_input = DataLoader(dataset=ds_input, batch_size=512, shuffle=False,
                          num_workers=8, prefetch_factor=512, drop_last=False, pin_memory=True)

    fp32_model = BraggNN(imgsz=11, fcsz=(16, 8, 4, 2)).eval()
    fp32_model.load_state_dict(torch.load('models/fc16_8_4_2-sz11.pth', map_location=torch.device('cpu')))

    test_input = torch.from_numpy(make_gaussian()).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        fp32_pred = fp32_model(test_input).cpu().numpy()
    print(f"fp32 prediction (pixel coords): {fp32_pred * 11}")

    example_inputs = (torch.randn(1, 1, 11, 11),)
    exported_model = torch.export.export(fp32_model, example_inputs).module()

    from torchao.quantization.pt2e.quantize_pt2e import (
        prepare_pt2e,
        convert_pt2e,
    )
    from backends.gemmini.quantizer import (
        GemminiQuantizer,
        get_symmetric_quantization_config,
    )

    quantizer = GemminiQuantizer().set_global(get_symmetric_quantization_config())
    prepared_model = prepare_pt2e(exported_model, quantizer)

    calibrate(prepared_model, dl_input)

    quantized_model = convert_pt2e(prepared_model)
    print(quantized_model)

    with torch.no_grad():
        int8_pred = quantized_model(test_input).cpu().numpy()
    print(f"int8  prediction (pixel coords): {int8_pred * 11}")

    #export_inputs = (test_input,)
    #with torch.no_grad():
    #    exported_program = export(quantized_model, export_inputs)

    #torch.export.save(exported_program, "models/int8_16_8_4_2-sz11.pth")
    #print("saved models/int8_16_8_4_2-sz11.pth")


if __name__ == "__main__":
    main()
