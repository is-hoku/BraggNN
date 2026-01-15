import numpy as np
import torch
from model import BraggNN

def make_gaussian():
    X_test = np.zeros((11, 11))
    x_cen, y_cen = 6.0, 5.0

    sig_x, sig_y = 0.6, 1.5
    for x in range(11):
        for y in range(11):
            X_test[y][x] = 1000*(np.exp(-(x-x_cen)*(x-x_cen)/2*sig_x - (y-y_cen)*(y-y_cen)/2*sig_y))
    
    X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())
    return torch.from_numpy(X_test[np.newaxis, np.newaxis].astype('float32'))

def main():
    model = BraggNN(imgsz=11, fcsz=(16, 8, 4, 2))
    mdl_fn = 'models/fc16_8_4_2-sz11.pth'
    model.load_state_dict(torch.load(mdl_fn, map_location=torch.device('cpu')))

    input_tensor = make_gaussian()
    with torch.no_grad():
        pred = model.forward(input_tensor).cpu().numpy()
        print(pred * 11)

if __name__ == "__main__":
    main()