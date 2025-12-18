import os
import shutil
import subprocess
import time
from utils.testMNNFromOnnx import TestModel
import onnx
from utils.extract_code_from_response import extract_code_from_response
import re
from pathlib import Path
from tqdm import tqdm

def get_connected_device():
    """
        get device infomation!
    """
    try:
        result = subprocess.run(
            ["adb", "devices"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        lines = result.stdout.strip().split('\n')
        devices = []
        
        for line in lines[1:]:  # skip: "List of devices attached"
            if line.strip() and 'device' in line:
                device_id = line.split('\t')[0]
                devices.append(device_id)
        
        return devices
    except subprocess.CalledProcessError:
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []
    
def restore_files(backup_dir: str |Path , 
                    target_folder: str|Path) -> bool:
    """Restore files in the target folder by copying backup files back."""

    # Make sure the target folder exists
    if not os.path.exists(target_folder):
        print(f"Error: Target folder {target_folder} does not exist.")
        return False

    # Traverse files in backup folder and restore to target folder
    file_list = os.listdir(backup_dir)
    if not file_list:
        print("No file to restore!")
        return True
    
    for backup_file in file_list:
        backup_path = os.path.join(backup_dir, backup_file)
        target_path = os.path.join(target_folder, backup_file)

        if os.path.exists(target_path):
            shutil.copy2(backup_path, target_path)
            os.sync()  # Flush filesystem cache to disk
            print(f"Restored: {backup_path} -> {target_path}")
            os.remove(backup_path)
        else:
            print(f"Warning: Target file {target_path} does not exist.")
            return False

    return True

    
def safe_write(src_file: str, dst_folder: str, code: str, backup_dir: Path) -> bool:
    """Copy file into dst_folder with automatic .bak backup if exists."""
    # clear before copy

    os.makedirs(backup_dir, exist_ok=True)
    dst_file = os.path.join(dst_folder, src_file)

    # If original cpp/hpp exists, make backup
    if os.path.exists(dst_file):
        backup_file = backup_dir/ src_file
        shutil.copy2(dst_file, backup_file)
        os.sync()  # Flush filesystem cache to disk
        # print(f"Backup created: {backup_file}")
    else:
        print(f"Error: no match file in target folder. Missing: {dst_file}")
        return False

    # Write file
    with open(os.path.join(dst_folder, src_file), "w") as f:
        f.write(code)

    print(f"Write new code: {src_file} -> {dst_file}")

    return True

def compile_project_withbar(op_name: str, llm: str, op_type: str, op_ctg: str,
                    target_folder: str = None,
                    backup_dir: str = None,
                    log_root: Path = None,
                    MNN_root: Path = None,
                    project_root:Path|None = None
                    ):
    """Run cmake and make to compile the project with a progress bar, hide terminal output and save logs."""
    
    # 1. Set log path
    log_dir = log_root / "compile_logs" / llm
    log_file = log_dir / op_type / op_ctg / (op_name + ".txt")
    os.makedirs(log_file.parent, exist_ok=True)
    
    # 2. Check if log already exists (skip logic)
    if os.path.exists(log_file):
        restore_files(backup_dir=backup_dir, target_folder=target_folder)
        print(f"Log exists for {op_name}. Restore op successfully! Skipping...")
        return False

    # Prepare regex to match CMake progress: [ 47%]
    progress_pattern = re.compile(r'\[\s*(\d+)%\]')

    try:
        # Open log file
        with open(log_file, "w") as log:
            # 3. Run register script
            os.chdir(MNN_root)
            subprocess.run(["python", "tools/script/register.py", "./"],
                           stdout=subprocess.DEVNULL, stderr=None)
            
            # 4. Clean and rebuild build directory
            build_dir = MNN_root / "build"
            if os.path.exists(build_dir):
                # print("Cleaning old build directory...")
                shutil.rmtree(build_dir)
        
            os.makedirs(build_dir, exist_ok=True)
            os.chdir(build_dir)
            
            # 5. Run CMake (fast step, no progress bar needed, log only)
            print(f"Configuring MNN for {op_name}...")
            subprocess.run(
                ["cmake", "..", "-DMNN_BUILD_CONVERTER=ON"], 
                check=True, 
                stdout=log, 
                stderr=log
            )

            # 6. Run Make (with tqdm progress bar)
            make_cmd = ["make", "-j8"]
            print(f"Compiling {op_name} ...")
            
            # Initialize progress bar
            pbar = tqdm(
                total=100,
                unit="%",
                desc="Building",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]'
            )
            
            # Start process using Popen
            process = subprocess.Popen(
                make_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout for logging and parsing
                text=True,
                bufsize=1
            )
            
            last_progress = 0
            
            # Read output line by line
            for line in process.stdout:
                # Write to log file
                log.write(line)
                
                # Regex match progress
                match = progress_pattern.search(line)
                if match:
                    current_progress = int(match.group(1))
                    if current_progress > last_progress:
                        pbar.update(current_progress - last_progress)
                        last_progress = current_progress
            
            # Wait for process to finish
            process.wait()
            
            # If successful, ensure progress bar reaches 100%
            if process.returncode == 0:
                if last_progress < 100:
                    pbar.update(100 - last_progress)
            
            pbar.close()

            # 7. Check Make return code
            if process.returncode != 0:
                # Manually raise exception to trigger restore logic in except block
                raise subprocess.CalledProcessError(process.returncode, make_cmd)

        # No exception → compile success
        return True

    except subprocess.CalledProcessError as e:
        # Compilation failed
        print(f"\n❌ Project compilation failed (Exit Code: {e.returncode}).")
        print(f"   Check details in: {log_file}")
        restore_files(backup_dir=backup_dir, target_folder=target_folder)
        print("Restore op successfully!")
        return False

    except Exception as e:
        # Catch other unexpected errors (e.g. permission issues, Ctrl+C, etc.)
        print(f"\n❌ An unexpected error occurred: {e}")
        restore_files(backup_dir=backup_dir, target_folder=target_folder)
        return False

def compile_project(op_name: str, llm: str, op_type: str,op_ctg:str,
                    target_folder: str = None,
                    backup_dir: str = None,
                    log_root:Path|None = None,
                    MNN_root:Path|None = None,
                    project_root:Path|None = None
                    ):
    """Run cmake and make to compile the project, hide terminal output and save logs."""
    log_dir = log_root / "compile_logs" / llm
    log_file = log_dir/op_type / op_ctg /(op_name + ".txt")
    os.makedirs(log_file.parent, exist_ok=True)
    # Set log file path
    # log_file = os.path.join(log_dir, compute_type + "_" + file_name + ".txt")
    
    if os.path.exists(log_file):
        restore_files(backup_dir=backup_dir,
                      target_folder=target_folder)
        print("Log file exists! Skip compile! Restore op successfully!")
        return False
    try:
        # Open log file
        with open(log_file, "w") as log:
            os.chdir(MNN_root)
            subprocess.run(["python", "tools/script/register.py", "./"], stdout=subprocess.DEVNULL, stderr=None)
            # Clean build directory to ensure no leftover files
            build_dir = MNN_root/"build"
            if os.path.exists(build_dir):
                print("Cleaning old build directory...")
                shutil.rmtree(build_dir)
        
            # Recreate the build directory
            os.makedirs(build_dir, exist_ok=True)
            os.chdir(build_dir)
            
            # Run cmake .. and write output to log
            subprocess.run(["cmake", "..", "-DMNN_BUILD_CONVERTER=ON"], check=True, stdout=log, stderr=log)
            # Run make -j8 and write output to log
            subprocess.run(["make", "-j8"], check=True, stdout=log, stderr=log)
            # subprocess.run(["make", "-j1"], cwd="build", check=True, stdout=log, stderr=log)
        # No exception → compile success
        return True

    except subprocess.CalledProcessError as e:
        # Compilation failed
        restore_files(backup_dir=backup_dir,
                      target_folder=target_folder)
        print("Project compilation failed. Please check the log file for details. Restore op successfully!")
        return False


def register_op_mnn(code_block: dict, backup_dir: str|Path ,
                    target_folder: str|Path) -> bool:
    
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)

    ok = False
    # Write files
    for key, value in code_block.items():
        if backup_dir is None:
            ok = safe_write(key, target_folder, value)
        else:
            ok = safe_write(key, target_folder, value, backup_dir)

    # If write failed, restore files
    if not ok:
        restore_files(target_folder=target_folder,backup_dir=backup_dir)

    return ok


def get_onnx_input(onnx_model_path):
    model = onnx.load(onnx_model_path)
    inputs = model.graph.input
    num_inputs = len(inputs)
    shapes = []

    for input in inputs:
        shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
        shape_str = 'x'.join(map(str, shape))
        shapes.append(shape_str)

    return num_inputs, shapes



def test_correctness(
                     MNN_root:Path,
                     onnx_model:Path,
                     mnn_model_path:Path,
                     compile:bool=False,
                     build_folder:str='build',
                     ):
    if not compile:
        return False,None,None
    
    num_inputs, shapes = get_onnx_input(onnx_model)

    os.chdir(MNN_root / build_folder)
    t = TestModel(str(onnx_model))
    result = t.Test(mnn_model_path=mnn_model_path)
    if 'test_success' in result.lower():
        correct = True
        print("Model output is correct!")
    else:
        correct = False
        if 'nullptr' in result.lower():
            print("The MNN model has no output.")
        elif 'mismatch' in result.lower():
            print("Output of the MNN model is different from ONNX model.")
    return correct, num_inputs, shapes

def test_performance_withbar(num_inputs, shapes,
                     project_root:Path ,
                     MNN_root:Path ,
                     correctness:bool=False,
                     build_folder:str="build_64",
                     ):
    if not correctness:
        return None
    base_dir = Path(MNN_root) / "project/android"
    build_dir = os.path.join(base_dir, build_folder)
    
    try:
        # Switch to base directory
        os.chdir(base_dir)
        
        # Completely clean the build directory
        if os.path.exists(build_dir):
            print("Cleaning old build directory...")
            shutil.rmtree(build_dir)
        
        # Recreate the build directory
        os.makedirs(build_dir, exist_ok=True)
        os.chdir(build_dir)
        
        # Execute build commands with error handling and output
        print("Starting build...")
        
        # Execute build script, keep stdout/stderr for debugging
        # result = subprocess.run(
        #     ["bash", base_dir / "build_64.sh"], 
        #     check=True, 
        #     stdout=subprocess.PIPE, 
        #     stderr=subprocess.PIPE,
        #     text=True
        # )
        # print("Build finished")
        make_cmd = ["bash", base_dir / "build_64.sh"]
            
        # Initialize progress bar
        pbar = tqdm(
            total=100,
            unit="%",
            desc="Building for Android",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]'
        )
        
        # Start process using Popen
        process = subprocess.Popen(
            make_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout for logging and parsing
            text=True,
            bufsize=1
        )
        
        last_progress = 0
        log_file = project_root/"logs"/"temp.log"
        progress_pattern = re.compile(r'\[\s*(\d+)%\]')
        with open(log_file, "w") as log:
            # Read output line by line
            for line in process.stdout:
                # Write to log file
                log.write(line)
                
                # Regex match progress
                match = progress_pattern.search(line)
                if match:
                    current_progress = int(match.group(1))
                    if current_progress > last_progress:
                        pbar.update(current_progress - last_progress)
                        last_progress = current_progress
            
            # Wait for process to finish
            process.wait()
            
            # If successful, ensure progress bar reaches 100%
            if process.returncode == 0:
                if last_progress < 100:
                    pbar.update(100 - last_progress)
            
            pbar.close()
            # Execute the update test script
        subprocess.run(
            ["bash", base_dir / "updateTest.sh"], 
            check=True, 
            stdout=subprocess.DEVNULL
        )
        devices = get_connected_device()
        print(f"New MNN architecture has been pushed to device(s): {devices}")
        
    except subprocess.CalledProcessError as e:
        print(f"Build failed, return code: {e.returncode}")
        if e.stdout:
            print("Standard Output:")
            print(e.stdout)
        if e.stderr:
            print("Error Output:")
            print(e.stderr)
        return None

    # Execute test command
    try:
        if num_inputs == 1:
            test_cmd = f"bash {MNN_root / "project/android/testCommon.sh"} ./MNNV2Basic.out model.mnn 100 0 0 4{shapes[0]}"
            test_result = subprocess.run(
                test_cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                check=True
            )
            # print("Test Output:")
            # print(test_result.stdout)
            return test_result.stdout
        else:
            # push input config into mobile devices
            subprocess.run(
                ["adb", "push" ,MNN_root / "build/onnx", "/data/local/tmp/MNN"], 
                check=True, 
                stdout=subprocess.DEVNULL
            )
            test_cmd = f"bash {MNN_root / "project/android/testCommon.sh"} ./ModuleBasic.out model.mnn onnx 0 0 100 4"
            test_result = subprocess.run(
                test_cmd, 
                shell=True, 
                # capture_output=True, 
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,#stderr → stdout
                text=True, 
                check=True
            )
            # print("Test Output:")
            # print(test_result.stdout)
            return test_result.stdout
            
    except subprocess.CalledProcessError as e:
        print(f"Test execution failed: {e}")
        if e.stderr:
            print("Test Error Output:")
            print(e.stderr)
        return None

    
def test_performance(num_inputs, shapes,
                     correctness:bool=False,
                     MNN_root:Path|None = None,
                     build_folder:str=None,
                     project_root:Path|None = None
                     ):
    if not correctness:
        return None
    base_dir = Path(MNN_root) / "project/android"
    build_dir = os.path.join(base_dir, build_folder)
    
    try:
        # Switch to base directory
        os.chdir(base_dir)
        
        # Completely clean the build directory
        if os.path.exists(build_dir):
            print("Cleaning old build directory...")
            shutil.rmtree(build_dir)
        
        # Recreate the build directory
        os.makedirs(build_dir, exist_ok=True)
        os.chdir(build_dir)
        
        # Execute build commands with error handling and output
        print("Starting build...")
        
        # Execute build script, keep stdout/stderr for debugging
        result = subprocess.run(
            ["bash", base_dir / "build_64.sh"], 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        print("Build finished")
        # Execute the update test script
        subprocess.run(
            ["bash", base_dir / "updateTest.sh"], 
            check=True, 
            stdout=subprocess.DEVNULL
        )
        devices = get_connected_device()
        print(f"New MNN architecture has been pushed to device(s): {devices}")
        
    except subprocess.CalledProcessError as e:
        print(f"Build failed, return code: {e.returncode}")
        if e.stdout:
            print("Standard Output:")
            print(e.stdout)
        if e.stderr:
            print("Error Output:")
            print(e.stderr)
        return None
    
    except Exception as e:
        print(f"Exception occurred during execution: {e}")
        return None
    
    # Execute test command
    try:
        if num_inputs == 1:
            test_cmd = f"bash /home/zeezou/MNN/project/android/testCommon.sh ./MNNV2Basic.out model.mnn 100 0 0 4{shapes[0]}"
            test_result = subprocess.run(
                test_cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                check=True
            )
            # print("Test Output:")
            # print(test_result.stdout)
            return test_result.stdout
        else:
            test_cmd = "bash /home/zeezou/MNN/project/android/testCommon.sh ./ModuleBasic.out model.mnn onnx 0 0 100 4"
            test_result = subprocess.run(
                test_cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                check=True
            )
            # print("Test Output:")
            # print(test_result.stdout)
            return test_result.stdout
            
    except subprocess.CalledProcessError as e:
        print(f"Test execution failed: {e}")
        if e.stderr:
            print("Test Error Output:")
            print(e.stderr)
        return None


def log_successful_operator(suffix, category, stage, file_name):
    log_dir = f'./recording/{suffix}/{category}'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"success_ops_stage{stage}.txt")
    with open(log_file, 'a') as f:
        f.write(f"{file_name}\n") 

def parse_operator_performance_2(log: str):
    """
    Parse MNN benchmark log and extract performance-related fields.

    Args:
        log (str): Raw log string

    Returns:
        dict: Parsed results
    """

    operator_performance = {
        "tensor_shape": None,   # not present in this log
        "precision": None,
        "memory": None,
        "runloop": 100,         # default
        "avg": None,
        "opsum": None,          # not present in this log
        "min": None,
        "max": None,
    }

    # precision=0 in main, 255
    m = re.search(r"precision\s*=\s*(\d+)", log)
    if m:
        operator_performance["precision"] = int(m.group(1))

    # memory=0 in main, 256
    m = re.search(r"memory\s*=\s*(\d+)", log)
    if m:
        operator_performance["memory"] = int(m.group(1))

    # Avg= 0.003360 ms, min= 0.001000 ms, max= 0.053000 ms
    m = re.search(
        r"Avg=\s*([\d.]+)\s*ms,\s*min=\s*([\d.]+)\s*ms,\s*max=\s*([\d.]+)\s*ms",
        log
    )
    if m:
        operator_performance["avg"] = float(m.group(1)) + " ms"
        operator_performance["min"] = float(m.group(2)) + " ms"
        operator_performance["max"] = float(m.group(3)) + " ms"

    return operator_performance

def parse_operator_performance(log_text: str):
    operator_performance = {
        "tensor_shape": None,
        "precision": None,
        "memory": None,
        "runloop": 100,
        "avg": None,
        "opsum": None,
        "min": None,
        "max": None,
    }
    
    # Tensor shape
    if log_text is None:
        return operator_performance
    

    m = re.search(r"Tensor shape\*\*:?\s*([0-9,\s]+)", log_text)
    if m:
        shape_nums = [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
        operator_performance["tensor_shape"] = shape_nums

    # precision
    m = re.search(r"precision:(\d+)", log_text)
    if m:
        operator_performance["precision"] = int(m.group(1))

    # memory
    m = re.search(r"memory:\s*(\d+)", log_text)
    if m:
        operator_performance["memory"] = int(m.group(1))

    # Run loop entries
    # runloop_lines = re.findall(
    #     r"(\w[\w_0-9]+).*?run 100 average cost ([0-9\.]+) ms.*?FlopsRate:\s*([0-9\.]+) %",
    #     log_text
    # )
    # for name, cost, flops in runloop_lines:
    #     operator_performance["runloop"].append({
    #         "op_name": name,
    #         "avg_cost_ms": float(cost),
    #         "flops_rate_percent": float(flops),
    #     })

    # Avg, OpSum, min, max
    m = re.search(
        r"Avg=\s*([0-9\.]+)\s*ms,\s*OpSum\s*=\s*([0-9\.]+)\s*ms\s*min=\s*([0-9\.]+)\s*ms,\s*max=\s*([0-9\.]+)\s*ms",
        log_text
    )
    if m:
        operator_performance["avg"] = m.group(1) + " ms"
        operator_performance["opsum"] = m.group(2) + " ms"
        operator_performance["min"] = m.group(3) + " ms"
        operator_performance["max"] = m.group(4) + " ms"
    else:
        m = re.search(
        r"Avg=\s*([\d.]+)\s*ms,\s*min=\s*([\d.]+)\s*ms,\s*max=\s*([\d.]+)\s*ms",
            log_text
        )
        if m:
            operator_performance["avg"] = m.group(1) + " ms"
            operator_performance["min"] = m.group(2) + " ms"
            operator_performance["max"] = m.group(3) + " ms"
    return operator_performance
if __name__=="__main__":
    txt = r"""
Use extra forward type: 0
inputDims: 
cpuIds: 
Open Model model.mnn
Can't open file:.tempcache
Load Cache file error.
CPU Group: [ 0  1  2  3 ], 300000 - 1804800
CPU Group: [ 4  5  6 ], 691200 - 2400000
CPU Group: [ 7 ], 806400 - 2515200
parsed /proc/cpuinfo Hardware = "zijin based Qualcomm Technologies, Inc. SM7325"
(last_midr & (CPUINFO_ARM_MIDR_IMPLEMENTER_MASK | CPUINFO_ARM_MIDR_PART_MASK))=0x 4100d410 in _getInfoArm, 1234 
The device supports: i8sdot:1, fp16:1, i8mm: 0, sve2: 0, sme2: 0
test_main, 303, cost time: 1.477000 ms
Session Info: memory use 3.000004 MB, flops is 0.515625 M, backendType is 13
===========> Session Resize Done.
===========> Session Start running...
Input size:524288
        **Tensor shape**: 8, 64, 32, 32, 
fileName.str().c_str()=s ./input_0.txt in _loadInputFromFile, 110 
output: output
precision:2, memory: 0, Run 100 time:
                                   output_raster_2      [Raster] run 100 average cost 0.001940 ms, 0.163 %, FlopsRate: 1.515 %
                                   output_raster_0      [Raster] run 100 average cost 0.174260 ms, 14.613 %, FlopsRate: 96.970 %
                                   output_raster_1      [Reduction] run 100 average cost 0.473580 ms, 39.714 %, FlopsRate: 1.515 %
Avg= 1.192490 ms, OpSum = 0.649780 ms min= 1.056000 ms, max= 1.787000 ms

Restored: /home/zeezou/MNN/backup_ops/CPUReduction.cpp -> /home/zeezou/MNN/source/backend/cpu/CPUReduction.cpp"""
    # d = get_connected_device()
    # print(d)
    performance = parse_operator_performance(txt)
    print(performance)
    pass
