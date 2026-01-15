import numpy as np
import torch
import torchao.quantization.pt2e
from model import BraggNN

def make_gaussian(imgsz=11, x_cen=6.0, y_cen=5.0, sig_x=0.6, sig_y=1.5, amp=1000.0, norm=True):
    x = np.arange(imgsz); y = np.arange(imgsz)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = np.exp(-((X-x_cen)**2)/(2*sig_x**2) - ((Y-y_cen)**2)/(2*sig_y**2))
    Z *= amp
    if norm:
        zmin, zmax = Z.min(), Z.max()
        if zmax > zmin: Z = (Z - zmin) / (zmax - zmin)
    return Z.astype(np.float32)

def main():
    example_inputs = (torch.from_numpy(make_gaussian()).unsqueeze(0).unsqueeze(0),)
    print("example_inputs: ", example_inputs)

    # fp32
    fp32_model = BraggNN(imgsz=11, fcsz=(16, 8, 4, 2)).eval()
    fp32_model.load_state_dict(torch.load('models/fc16_8_4_2-sz11.pth', map_location=torch.device('cpu')))
    with torch.no_grad():
        pred = fp32_model.forward(*example_inputs).cpu().numpy()
        print("fp32: ", pred * 11)

    # int8
    int8_model = torch.export.load('models/int8_16_8_4_2-sz11.pth').module()
    #int8_model = torch.load(
    #    'models/int8_16_8_4_2-sz11.pth',
    #    map_location=torch.device('cpu'),
    #).eval()

    with torch.no_grad():
        pred = int8_model.forward(*example_inputs).cpu().numpy()
        print("int8: ", pred * 11)


if __name__ == "__main__":
    main()