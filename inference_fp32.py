import numpy as np
import torch
import ctypes
import mmap
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import BraggNN
from dataset import BraggNNDataset

NUM_PATCHES = 10
PSZ = 11


def _make_rdtsc():
    """Create a callable that executes the x86 RDTSC instruction."""
    code = bytes([
        0x0F, 0x31,             # rdtsc          (EDX:EAX = TSC)
        0x48, 0xC1, 0xE2, 0x20, # shl rdx, 32
        0x48, 0x09, 0xD0,       # or  rax, rdx
        0xC3,                   # ret
    ])
    buf = mmap.mmap(-1, len(code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
    buf.write(code)
    func_type = ctypes.CFUNCTYPE(ctypes.c_uint64)
    c_buf = ctypes.c_char.from_buffer(buf)
    func = func_type(ctypes.addressof(c_buf))
    # Keep references alive so the mmap buffer is not garbage-collected
    func._prevent_gc = (buf, c_buf)
    return func


rdtsc = _make_rdtsc()


def _load_patch(ds, idx):
    """Load a single patch from the dataset, returning (input_tensor, label)."""
    feat, label = ds[int(idx)]
    if isinstance(feat, np.ndarray):
        feat = torch.from_numpy(feat)
    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)
    return feat.unsqueeze(0), label  # (1, 1, 11, 11), (2,)


def main():
    # Load FP32 model
    model = BraggNN(imgsz=PSZ, fcsz=(16, 8, 4, 2))
    mdl_fn = 'models/fc16_8_4_2-sz11.pth'
    model.load_state_dict(torch.load(mdl_fn, map_location=torch.device('cpu')))
    model.eval()

    # Load validation dataset (same as export_input_patches.py)
    ds = BraggNNDataset(psz=PSZ, rnd_shift=0, use='validation')
    indices = np.linspace(0, len(ds) - 1, NUM_PATCHES, dtype=int)

    print("==============================================")
    print("BraggNN FP32 inference through PyTorch")
    print("==============================================")
    print(f"\nInput shape: [1, 1, {PSZ}, {PSZ}]")
    print(f"Model: {mdl_fn}")
    print(f"Running {NUM_PATCHES + 1} inferences (1 warmup + {NUM_PATCHES} measured)")

    total_cycles = 0
    total_x_error = 0.0
    total_y_error = 0.0

    with torch.no_grad():
        # Warmup: drop first run to warm caches
        warmup_idx = indices[len(indices) // 2]  # use middle patch like C uses index 5
        input_tensor, label = _load_patch(ds, warmup_idx)

        print("\n--- Warmup (patch 0) ---")
        c0 = rdtsc()
        output = model(input_tensor)
        pred = output.numpy()[0]
        pred_px = pred * PSZ
        c1 = rdtsc()
        elapsed = c1 - c0
        true_px = label.numpy() * PSZ
        err_x = pred_px[0] - true_px[0]
        err_y = pred_px[1] - true_px[1]
        print(f"cycles: {elapsed}")
        print(f"pred: ({pred_px[0]:.4f}, {pred_px[1]:.4f}), "
              f"actual: ({true_px[0]:.4f}, {true_px[1]:.4f}), "
              f"error: ({err_x:.4f}, {err_y:.4f})")
        print("Warmup done")

        # Measured iterations
        for i, idx in enumerate(indices):
            input_tensor, label = _load_patch(ds, idx)

            print(f"\n--- Inference {i + 1}/{NUM_PATCHES} ---")
            c0 = rdtsc()
            output = model(input_tensor)
            pred = output.numpy()[0]
            pred_px = pred * PSZ
            c1 = rdtsc()
            elapsed = c1 - c0

            total_cycles += elapsed
            true_px = label.numpy() * PSZ
            err_x = pred_px[0] - true_px[0]
            err_y = pred_px[1] - true_px[1]
            total_x_error += abs(err_x)
            total_y_error += abs(err_y)

            print(f"cycles: {elapsed}")
            print(f"pred: ({pred_px[0]:.4f}, {pred_px[1]:.4f}), "
                  f"actual: ({true_px[0]:.4f}, {true_px[1]:.4f}), "
                  f"error: ({err_x:.4f}, {err_y:.4f})")

    print(f"\n==============================================")
    print(f"Avg cycles over {NUM_PATCHES} runs: {total_cycles // NUM_PATCHES}")
    print(f"Avg error over {NUM_PATCHES} runs: "
          f"({total_x_error / NUM_PATCHES:.4f}, {total_y_error / NUM_PATCHES:.4f})")
    print("==============================================")
    print("BraggNN inference completed successfully")
    print("==============================================" )


if __name__ == "__main__":
    main()
