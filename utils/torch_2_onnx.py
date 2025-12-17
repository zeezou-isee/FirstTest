import os
import importlib.util
import torch
import onnxruntime as ort
import numpy as np
import inspect
import sys

def load_model_from_path(py_path):
    """
    dynamically import python file as module
    """
    module_name = os.path.splitext(os.path.basename(py_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_torch_as_onnx(model, inputs, save_onnx_path):
    """
    export torch model as onnx model
    """
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    input_names = [f"input{i+1}" for i in range(len(inputs))]

    torch.onnx.export(
        model,
        tuple(inputs),
        save_onnx_path,
        input_names=input_names,
        output_names=["output"],
        dynamo=False
    )


def check_transform(onnx_model_path, model, example_inputs):
    """
    compare outputs between PyTorch and ONNX Runtime
    """
    model.eval()

    if not isinstance(example_inputs, (list, tuple)):
        example_inputs = [example_inputs]
    example_inputs = tuple(example_inputs)

    with torch.no_grad():
        torch_out = model(*example_inputs)
    torch_out_np = torch_out.detach().cpu().numpy() if torch.is_tensor(torch_out) else torch_out

    ort_session = ort.InferenceSession(onnx_model_path)
    ort_inputs = {}
    for i, tensor in enumerate(example_inputs):
        ort_inputs[ort_session.get_inputs()[i].name] = tensor.detach().cpu().numpy()
    ort_out = ort_session.run(None, ort_inputs)[0]

    difference = np.abs(torch_out_np - ort_out)

    difference = (difference - 1e-2 * np.abs(torch_out_np)).max()

    return difference


def test_single_script(py_file, model_folder, onnx_folder):
    """
    1. Export torch model as onnx.
    2. Test transform correctness.
    """
    module = load_model_from_path(py_file)

    # Find the first nn.Module subclass
    model_class = None
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, torch.nn.Module) and obj is not torch.nn.Module:
            model_class = obj
            break
    if model_class is None:
        raise ValueError(f"No nn.Module subclass found in {py_file}")

    # Handle constructor parameters
    init_args = []
    if hasattr(module, "get_init_inputs"):
        val = module.get_init_inputs()
        if isinstance(val, (list, tuple)):
            init_args = list(val)
        else:
            init_args = [val]

    model = model_class(*init_args)

    # Get example inputs
    if not hasattr(module, "get_inputs"):
        raise ValueError(f"{py_file} does not define get_inputs()")
    example_inputs = module.get_inputs()

    # Prepare ONNX file path preserving original structure in the new folder
    relative_path = os.path.relpath(py_file, model_folder)  # Get relative path from model_folder
    onnx_path = os.path.splitext(relative_path)[0] + ".onnx"  # Change extension to .onnx

    # Build the new target folder for ONNX models
    onnx_dir = os.path.join(onnx_folder, os.path.dirname(onnx_path))

    # Create necessary directories in the new folder
    os.makedirs(onnx_dir, exist_ok=True)

    # Export model to ONNX
    export_torch_as_onnx(model, example_inputs, os.path.join(onnx_folder, onnx_path))

    # Compare PyTorch and ONNX output
    # diff = check_transform(os.path.join(onnx_folder, onnx_path), model, example_inputs)
    # result = diff < 1e-2

    return {
        "model_script": os.path.basename(py_file),
        "onnx_file": os.path.join(onnx_folder, onnx_path),
        # "difference": diff,
        # "pass": result,
    }


def traverse_and_process_models(model_folder, onnx_folder):
    """
    Recursively traverse the folder, process all Python model scripts, 
    and convert them to ONNX, saving to the specified ONNX folder.
    """
    results = []
    this_file = os.path.basename(__file__)

    for dirpath, dirnames, filenames in os.walk(model_folder):
        for file in filenames:
            # Skip non-Python files and the script itself
            if not file.endswith(".py") or file == this_file:
                continue

            py_path = os.path.join(dirpath, file)

            try:
                info = test_single_script(py_path, model_folder, onnx_folder)  # Pass the new folder path
                results.append(info)
                print(f"[✓] {file}")
            except Exception as e:
                print(f"[✗] {file}: ERROR - {e}")

    print("\n===== Summary =====")
    for r in results:
        print(f"{r['model_script']:<40}")


    
def main(model_folder):
    onnx_folder = '/home/zeezou/MNN/onnx_dataset_test/model_60'
    traverse_and_process_models(model_folder, onnx_folder)  # 传递新文件夹路径


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python torch_2_onnx.py <models_folder>")
        sys.exit(1)
    models_folder = sys.argv[1]
    main(models_folder)
