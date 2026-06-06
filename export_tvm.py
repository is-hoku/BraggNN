import numpy as np
import torch
from torch.export import export
from model import BraggNN
from torch.utils.data import DataLoader
from dataset import BraggNNDataset


import tvm
import tvm.contrib.gemmini.build
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
import tvm.relax.backend.contrib.gemmini
import tvm.contrib.gemmini.build
from tvm.relax.backend.pattern_registry import get_patterns_with_prefix
from tvm.relax.transform import (
    ConvertToDataflow,
    Normalize,
    CanonicalizeBindings,
    FoldConstant,
    FoldRedundantBroadcastTo,
    FoldPermuteDims,
    DecomposeOpsForInference,
    ConvertLayout,
    EliminateCommonSubexpr,
    DeadCodeElimination,
    FuseOpsByPattern,
    MergeCompositeFunctions,
    FuseOps,
    FuseTIR,
    LegalizeOps,
    StaticPlanBlockMemory,
    RunCodegen,
    CallTIRRewrite,
    VMShapeLower,
    LowerRuntimeBuiltin,
    LowerAllocTensor,
    KillAfterLastUse,
    AttachGlobalSymbol,
    AnnotateTIROpPattern,
)


#def npz_to_carray(params_path, params):
#    with open(params_path, "w") as f:
#        f.write("#pragma once\n\n")
#        f.write("#include <stdint.h>\n\n")
#
#        for i, p in enumerate(params["main"]):
#            arr = p.numpy()
#            dtype = arr.dtype
#
#            if dtype == np.float32:
#                c_type = "float"
#                fmt = lambda x: f"{x:.8f}f"
#            elif dtype == np.float64:
#                c_type = "double"
#                fmt = lambda x: f"{x:.16f}"
#            elif dtype == np.int32:
#                c_type = "int32_t"
#                fmt = lambda x: str(int(x))
#            elif dtype == np.int64:
#                c_type = "int64_t"
#                fmt = lambda x: str(int(x))
#            elif dtype == np.int8:
#                c_type = "int8_t"
#                fmt = lambda x: str(int(x))
#            else:
#                c_type = "float"
#                fmt = lambda x: f"{x:.8f}f"
#
#            shape_bracket = "".join(f"[{d}]" for d in arr.shape)
#            shape_comment = "x".join(str(d) for d in arr.shape)
#
#            f.write(f"// shape: ({shape_comment})\n")
#            f.write(f"static const {c_type} p_{i}{shape_bracket} = ")
#
#            def write_array(f, arr, fmt, depth=0):
#                if arr.ndim == 1:
#                    f.write("{")
#                    f.write(", ".join(fmt(v) for v in arr))
#                    f.write("}")
#                else:
#                    indent = "    " * depth
#                    f.write("{\n")
#                    for j, sub in enumerate(arr):
#                        f.write(f"{indent}    ")
#                        write_array(f, sub, fmt, depth + 1)
#                        if j < len(arr) - 1:
#                            f.write(",")
#                        f.write("\n")
#                    f.write(f"{indent}}}")
#
#            write_array(f, arr, fmt)
#            f.write(";\n\n")


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
            # Process each sample individually since exported model expects batch_size=1
            for i in range(image.shape[0]):
                model(image[i:i+1])

def main():
    ds_input = BraggNNDataset(psz=11, rnd_shift=0, use='validation')
    dl_input = DataLoader(dataset=ds_input, batch_size=512, shuffle=False, \
                          num_workers=8, prefetch_factor=512, drop_last=False, pin_memory=True)
    # For torh 1.6.0
    #dl_input = DataLoader(dataset=ds_input, batch_size=512, shuffle=False, \
    #                      num_workers=8, drop_last=False, pin_memory=True)
    fp32_model = BraggNN(imgsz=11, fcsz=(16, 8, 4, 2)).eval()
    fp32_model.load_state_dict(torch.load('models/fc16_8_4_2-sz11.pth', map_location=torch.device('cpu')))

    example_inputs = (torch.randn(1, 1, 11, 11),)
    exported_model = torch.export.export(fp32_model, example_inputs).module()

    from torchao.quantization.pt2e.quantize_pt2e import (
        prepare_pt2e,
        convert_pt2e,
    )

    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        get_symmetric_quantization_config,
        XNNPACKQuantizer,
    )

    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
    prepared_model = prepare_pt2e(exported_model, quantizer)

    calibrate(prepared_model, dl_input)

    quantized_model = convert_pt2e(prepared_model)
    example_inputs = (torch.from_numpy(make_gaussian()).unsqueeze(0).unsqueeze(0),)

    with torch.no_grad():
        exported_program = export(quantized_model, example_inputs)
        mod = from_exported_program(
            exported_program,
            keep_params_as_input=False,
            unwrap_unit_return_tuple=True,
        )

    # Export PyTorch into Relax IR
    #mod, params = relax.frontend.detach_params(mod)

    #params_path = "gemmini_out/params.h"
    #npz_to_carray(params_path, params)
    #print(f"Saved parameters to: {params_path}")

    print("After exporting:")
    #mod.show()
    print(mod.script())

    has_gemmini_codegen = tvm.get_global_func("relax.ext.gemmini", True)

    patterns = get_patterns_with_prefix("gemmini")
    print("Registered patterns:", [p.name for p in patterns])

    pipeline = tvm.transform.Sequential([
        ConvertToDataflow(),
        Normalize(),
        FoldConstant(),
        DecomposeOpsForInference(),
        FoldRedundantBroadcastTo(),
        CanonicalizeBindings(),
        DeadCodeElimination(),

        ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]}),
        FoldPermuteDims(),
        FuseOpsByPattern(patterns, annotate_codegen=True, bind_constants=False),
        MergeCompositeFunctions(),
        RunCodegen(),

        LegalizeOps(),
        AnnotateTIROpPattern(),
        FuseOps(),
        FuseTIR(),
        FoldConstant(),
        DeadCodeElimination(),

    ])

    with tvm.transform.PassContext(opt_level=3):
        mod = pipeline(mod)
    print("After pipeline:")
    print(mod.script)
    print(mod)

    #target = tvm.target.Target({"kind": "llvm", "mtriple": "riscv64-unknown-linux-gnu"})
    target = tvm.target.Target("c")
    ex = tvm.compile(mod, target)
    ex.export_library("braggnn.so", cc="riscv64-unknown-linux-gnu-gcc")
    print("All done")


if __name__ == "__main__":
    main()
