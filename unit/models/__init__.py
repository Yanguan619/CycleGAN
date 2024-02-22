from .cycle_gan import CycleGANModel
from .munit import MUNITModel
from .funit import FUNITModel


# def model2onnx(model, input_size):
#     import torch
#     import onnx
#     from onnx import shape_inference

#     img = torch.rand(*input_size)
#     torch.onnx.export(
#         model=model,
#         args=img,
#         f="model.onnx",
#         input_names=["image"],
#         output_names=["feature_map"],
#     )
#     onnx.save(onnx.shape_inference.infer_shapes(onnx.load("model.onnx")), "model.onnx")
