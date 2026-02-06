import numpy as np
import onnx
import onnxruntime as ort

def generate_dummy_input():
    psz = 11  # BRAGGNN_PATCH_SIZE
    input_data = np.zeros(psz * psz, dtype=np.float32)

    center_x = 5.5
    center_y = 5.5
    sigma = 2.0

    for i in range(psz):
        for j in range(psz):
            dx = float(j) - center_x
            dy = float(i) - center_y
            dist_sq = dx * dx + dy * dy
            input_data[i * psz + j] = np.exp(-dist_sq / (2.0 * sigma * sigma))

    # Add some noise
    for i in range(psz * psz):
        input_data[i] += (i % 7) / 70.0

    # Normalize to [0, 1]
    min_val = input_data.min()
    max_val = input_data.max()

    if max_val > min_val:
        input_data = (input_data - min_val) / (max_val - min_val)

    print("Generated dummy input data (11x11 patch with Gaussian-like peak)")

    # Reshape to (1, 1, 11, 11) for model input
    return input_data.reshape(1, 1, psz, psz)

def main():
    model_path = "onnx/braggnn_int8_qlinear_opset10.onnx"
    model = onnx.load(model_path)
    model.opset_import[0].version = 12

    ort_sess = ort.InferenceSession(model.SerializeToString())
    input_data = generate_dummy_input()
    outputs = ort_sess.run(None, {'input': input_data})
    pred = outputs[0]
    print("pred (output): (%f, %f)" % (pred[0][0], pred[0][1]))
    print("pred (pixel): (%f, %f)" % (pred[0][0] * 11, pred[0][1] * 11))

if __name__ == "__main__":
    main()

