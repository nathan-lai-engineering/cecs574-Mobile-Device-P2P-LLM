import torch
import torch.nn as nn
import sys
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static
from onnxruntime.quantization import MatMulWeight4Quantizer
from onnxruntime.quantization.quant_utils import load_model_with_shape_infer
import os
import shutil

# Project root is 3 levels above this file:
#   system_pipeline/quantization/quantize.py → project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def onnx_quantize_model(model_path: str):
    """
    Quantize the model, so that it can be run on mobile devices with smaller memory footprint
    """
    model_path = Path(model_path).resolve()
    model_path_dir = str(model_path.parent)
    quantized_model_path = model_path.with_name(model_path.stem + "_quant").with_suffix(model_path.suffix)
    quantized_model_name = quantized_model_path.stem
    quantize_dynamic(model_path, quantized_model_path, weight_type=QuantType.QInt8, use_external_data_format=True)
    model_path.unlink()
    link_external_data(model_path_dir, quantized_model_name)
    return quantized_model_path


def link_external_data(model_path: str, model_name: str):
    model_path = os.path.abspath(model_path)
    data_filename = model_name + ".onnx.data"
    destination_dir = os.path.join(model_path, data_filename)

    # Newer onnxruntime writes the .data file alongside the output .onnx — already correct.
    if os.path.exists(destination_dir):
        print(f"External data already in model directory: {destination_dir}")
        return

    # Older onnxruntime writes the .data file to cwd (project root). Move it.
    project_dir = str(_PROJECT_ROOT)
    original_dir = os.path.join(project_dir, data_filename)
    print(f"Looking for external data at: {original_dir}")
    print(f"Destination: {destination_dir}")

    if os.path.exists(original_dir):
        print(f"Found external protobuf data for {model_name}, moving to model directory")
        shutil.move(original_dir, destination_dir)
    else:
        raise RuntimeError(f"External data file not found at {destination_dir} or {original_dir}")








