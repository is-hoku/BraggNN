#!/usr/bin/env python
import argparse, numpy as np, torch, torch.nn as nn
from model import BraggNN

# FX APIs
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, fuse_fx

# Try new APIs (PyTorch ≥ ~2.1). Fallback to legacy on ImportError.
try:
    from torch.ao.quantization import get_default_qconfig_mapping, get_default_backend_config
    NEW_API = True
except ImportError:
    from torch.ao.quantization import get_default_qconfig, QConfigMapping
    NEW_API = False

torch.backends.quantized.engine = "fbgemm"  # for x86

# make a test input data
def make_gaussian(imgsz, x_cen=6.0, y_cen=5.0, sig_x=0.6, sig_y=1.5, amp=1000.0, norm=True):
    x = np.arange(imgsz); y = np.arange(imgsz)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = np.exp(-((X-x_cen)**2)/(2*sig_x**2) - ((Y-y_cen)**2)/(2*sig_y**2))
    Z *= amp
    if norm:
        zmin, zmax = Z.min(), Z.max()
        if zmax > zmin: Z = (Z - zmin) / (zmax - zmin)
    return Z.astype(np.float32)

# 
def predict_xy(model: nn.Module, img: np.ndarray):
    assert img.ndim == 2
    imgsz = img.shape[0]
    x = torch.from_numpy(img[None, None])  # 1x1xH xW
    with torch.inference_mode():
        out = model(x).squeeze().cpu().numpy()  # [x_norm, y_norm]
    return out * (imgsz - 1)

def load_state_dict_safely(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

# INT8 PTQ
def quantize_bragg(model_fp32: nn.Module, imgsz: int, calib_iters: int = 128, seed: int = 0):
    model_fp32.eval().cpu() # make sure this is in the inference mode
    fused = fuse_fx(model_fp32)  # no-op if nothing to fuse

    example = torch.randn(1, 1, imgsz, imgsz)

    if NEW_API:
        qmap = get_default_qconfig_mapping("fbgemm")
        backend_config = get_default_backend_config("fbgemm")
        prepared = prepare_fx(fused, qmap, example_inputs=example, backend_config=backend_config)
    else:
        # Legacy path (works on older torch); may show a benign fixed-qparams warning.
        qconfig = get_default_qconfig("fbgemm")  # activations and weights to int8
        qmap = QConfigMapping().set_global(qconfig)  # global: conv, linear, etc
        prepared = prepare_fx(fused, qmap, example_inputs=example) # quantize prep.

    # Calibrate
    rng = np.random.default_rng(seed)
    with torch.inference_mode():
        for _ in range(calib_iters):
            xc = float(rng.uniform(0.5, imgsz-1.5))
            yc = float(rng.uniform(0.5, imgsz-1.5))
            img = make_gaussian(imgsz, xc, yc)
            prepared(torch.from_numpy(img[None, None]))

    #  replace FP32 ops with quantized int8 counterparts
    if NEW_API:
        quantized = convert_fx(prepared, backend_config=backend_config).eval()
    else:
        quantized = convert_fx(prepared).eval()

    return quantized

# just for convenience
def plot_example(img, xy_pred, xy_true=None, title=None):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plot."); return
    H, W = img.shape
    plt.figure()
    plt.imshow(img, origin="upper", cmap="viridis", extent=[0, W-1, H-1, 0])
    px, py = xy_pred
    plt.scatter([px], [py], marker="x", s=80, label="pred")
    if xy_true is not None:
        tx, ty = xy_true
        plt.scatter([tx], [ty], marker="+", s=80, label="true")
    plt.legend(); 
    if title: plt.title(title)
    plt.tight_layout(); plt.show()


def main():
    ap = argparse.ArgumentParser(description="BraggNN FP32/INT8 inference (PTQ FX, version-compatible)")
    ap.add_argument("--imgsz", type=int, default=11)
    ap.add_argument("--x_cen", type=float, default=6.0)
    ap.add_argument("--y_cen", type=float, default=5.0)
    ap.add_argument("--sig_x", type=float, default=0.6)
    ap.add_argument("--sig_y", type=float, default=1.5)
    ap.add_argument("--amp",   type=float, default=1000.0)
    ap.add_argument("--no-norm", action="store_true")
    ap.add_argument("--weights", type=str, default="models/fc16_8_4_2-sz11.pth")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--quantize", action="store_true", help="enable INT8 PTQ (weights + activations)")
    ap.add_argument("--calib-iters", type=int, default=128)
    ap.add_argument("--save-int8", type=str, default="")
    args = ap.parse_args()

    model = BraggNN(imgsz=args.imgsz, fcsz=(16, 8, 4, 2))
    load_state_dict_safely(model, args.weights)
    model.eval()

    if args.quantize:
        model = quantize_bragg(model, imgsz=args.imgsz, calib_iters=args.calib_iters)
        print(model)
        if args.save_int8:
            torch.save(model.state_dict(), args.save_int8)

        int8_output_path = "onnx/braggnn_int8_opset13_ptqfwd.onnx"
        example_input = make_gaussian(
            imgsz=args.imgsz,
            x_cen=args.x_cen, y_cen=args.y_cen,
            sig_x=args.sig_x, sig_y=args.sig_y,
            amp=args.amp, norm=not args.no_norm
        )
        # Convert to tensor with shape (batch, channel, H, W)
        example_input = torch.from_numpy(example_input[None, None])

        torch.onnx.export(
            model,
            example_input,
            int8_output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=13, # For use old opsets
            #opset_version=18,
            export_params=True,
            do_constant_folding=True,
            dynamo=False, # For use old APIs
        )

    img = make_gaussian(
        imgsz=args.imgsz,
        x_cen=args.x_cen, y_cen=args.y_cen,
        sig_x=args.sig_x, sig_y=args.sig_y,
        amp=args.amp, norm=not args.no_norm
    )
    xy_pred = predict_xy(model, img)
    xerr, yerr = xy_pred[0]-args.x_cen, xy_pred[1]-args.y_cen

    print(f"torch      : {torch.__version__}  (new_api={NEW_API})")
    print(f"mode       : {'INT8' if args.quantize else 'FP32'}")
    print(f"pred(px)   : ({xy_pred[0]:.4f}, {xy_pred[1]:.4f})")
    print(f"true(px)   : ({args.x_cen:.4f}, {args.y_cen:.4f})")
    print(f"error      : ({xerr:.6f}, {yerr:.6f})")

    if args.plot:
        plot_example(img, xy_pred, (args.x_cen, args.y_cen), title="Gaussian + Pred vs True")

if __name__ == "__main__":
    main()
