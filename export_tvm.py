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
import tvm.contrib.gemmini.build
from tvm.relax.backend.pattern_registry import get_patterns_with_prefix
from tvm.relax.transform import (
    ConvertToDataflow,
    Normalize,
    CanonicalizeBindings,
    FoldConstant,
    FoldRedundantBroadcastTo,
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
)
from tvm.script import relax as R

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
            keep_params_as_input = True,
            unwrap_unit_return_tuple = True,
        )

    # Export PyTorch into Relax IR
    mod, params = relax.frontend.detach_params(mod)
    print("After exporting:")
    mod.show()

    has_gemmini_codegen = tvm.get_global_func("relax.ext.gemmini", True)
    has_gemmini = has_gemmini_codegen #and has_gemmini_runtime

    target = tvm.target.Target("c")

    patterns = get_patterns_with_prefix("gemmini")
    print("Registered patterns:", [p.name for p in patterns])

    pipeline = tvm.transform.Sequential([
        ConvertToDataflow(),
        Normalize(),
        FoldConstant(),
        DecomposeOpsForInference(),
        FoldRedundantBroadcastTo(),
        #EliminateCommonSubexpr(),
        CanonicalizeBindings(),
        DeadCodeElimination(),

        FuseOpsByPattern(patterns, annotate_codegen=True),
        ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]}),

        #MergeCompositeFunctions(),
        #FuseOps(),
        #FuseTIR(),
        #LegalizeOps(),
        #DeadCodeElimination(),
        #StaticPlanBlockMemory(),

        RunCodegen(),

        #CallTIRRewrite(),
        #VMShapeLower(),
        #LowerRuntimeBuiltin(),
        #LowerAllocTensor(),
        #KillAfterLastUse(),
        #AttachGlobalSymbol(),
    ])

    with tvm.transform.PassContext(opt_level=3):
        mod = pipeline(mod)
    print("After pipeline:")
    print(mod)

if __name__ == "__main__":
    main()
