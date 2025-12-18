import os
from pathlib import Path
import shutil
import argparse
import subprocess


def build_MNN_First_time(MNN_root:Path):
    try :
        subprocess.run(["python", "tools/script/register.py", "./"],
                           stdout=subprocess.DEVNULL, stderr=None)
        os.chdir(MNN_root)
        build_dir = MNN_root / "build"
        if os.path.exists(build_dir):
            print("Cleaning old build directory...")
            shutil.rmtree(build_dir)
        os.makedirs(build_dir, exist_ok=True)
        # build 
        os.chdir(build_dir)
        print("Cmake...")
        result = subprocess.run(
            ["cmake", "..", "-DMNN_BUILD_CONVERTER=ON"], 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        print("Build directory...")
        result = subprocess.run(
            ["make", "-j8"], 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        # build for android
        base_dir = MNN_root/ "project/android"
        os.chdir(base_dir)
        android_build_dir = base_dir / "build_64"
        if os.path.exists(android_build_dir):
            print("Cleaning old build directory...")
            shutil.rmtree(android_build_dir)
        os.makedirs(android_build_dir,exist_ok=True)
        os.chdir(android_build_dir)
        print("Build directory for Android...")
        result = subprocess.run(
            ["bash", "../build_64.sh"], 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        print("Push tools to Android...")
        result = subprocess.run(
            ["bash", "../updateTest_MNN.sh"], 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"Updata file finished, check {base_dir/"updateTest_MNN.sh"} for details.")
        # rename build     
        if os.path.exists(MNN_root/'build_MNN'):
            shutil.rmtree(MNN_root/'build_MNN')   
        result = subprocess.run(
            ["mv", MNN_root/'build',MNN_root/'build_MNN'], 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        print("Build MNN success!")
        return True
    except Exception as e:
        print("Build MNN failed, please clean this repo and build manually.")
        return False


if __name__ =="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--MNN-root", type=str,default="/home/zeezou/MNN",required=True,help="absolute path to MNN project")
    args = parser.parse_args()

    build_MNN_First_time(Path(args.MNN_root))

