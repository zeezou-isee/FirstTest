import os
from pathlib import Path
import sys
import shutil
sys.path.append("/home/zeezou/MNN/MobileDeviceKernelBench")
from utils.utils import (
                            restore_files,
                            register_op_mnn,
                            compile_project,
                            test_correctness,
                            test_performance,
                            parse_operator_performance,
                            parse_operator_performance_2,
                        )
from scripts.BasicConfig import OpConfig
from utils.extract_code_from_response import extract_code_from_response
from prompt.prompt_generator import generate_prompt_with_py_model,read_file
from prompt.utils import query_llm_openrouter,query_llm_alicloude
import subprocess
import argparse

    

def gen_and_test_single_op_performance_MNN(cfg:OpConfig):
    """
        To use this function, build MNN first:
            1. mkdir build_MNN
            2. cmake .. -DMNN_BUILD_CONVERTER=ON
            3. make -j8
        To test performance on ANDROID phone, build for Android:
            1. cd project/android
            2. mkdir build_64_MNN
            3. bash ../build_64.sh
    """
    # init info
    compile_info = "fail"
    correctness= "fail"
    performance = ""

    operator_evaluation_info = {
        "name":cfg.op_name,
        "type":cfg.op_type,
        "category":cfg.op_ctg,
        "generated":cfg.cfg_LLM,
        "register":"pass",
        "compile":"",
        "correctness": "",
        "performance": "",
    }

    # 1.prepare MNN for test
    # You should run [python /home/zeezou/MNN/MobileDeviceKernelBench/scripts/buil_MNN_First_time.py ./] first
       
    # 2. test correctness of op
    print("---"*20,f"Start Test Correctness operator: {cfg.op_name}","---"*20)
    op_mnn_path = Path(*[part for part in cfg.op_mnn_path.parts if part != cfg.cfg_LLM])
    correct, num_inputs, shapes = test_correctness(compile=True,
                                                   MNN_root=cfg.MNN_root,
                                                    onnx_model=cfg.op_onnx_path,
                                                    build_folder="build_MNN",
                                                    mnn_model_path=op_mnn_path)
    # export LD_LIBRARY_PATH="$PWD/tools/converter:$PWD:$LD_LIBRARY_PATH" ./MNNConvert -f ONNX --bizCode MNN --modelFile /home/zeezou/MNN/MobileDeviceKernelBench/source/onnx_dataset_test/model_60/Atomic/reduction/ReduceSum.onnx --MNNModel /home/zeezou/MNN/MobileDeviceKernelBench/source/mnn_models_original/model_60/Atomic/reduction/ReduceSum.onnx --keepInputFormat=1 --testdir onnx

    # 3. test performance of  op    
    #   3.1 push test model to mobile phone
    if correct:
        print("Operator correctness varification by MNN success!")
        correctness = "Model output is correct!"

        print("adb push model to phone")
        subprocess.run(
                ["adb","push",str(op_mnn_path), "/data/local/tmp/MNN_original/model.mnn"],
                check=True, 
                stdout=subprocess.DEVNULL
            )
    #   3.2 build MNN original for android and test performance
    print("Test the performance of operator in MNN")
    # if fail here, plwase build this for android under :project/android/build_64_MNN
    sh_path = cfg.MNN_root / "project/android" / "testCommon_MNN.sh"
    if num_inputs == 1:
            test_cmd = f"bash {sh_path} ./MNNV2Basic.out model.mnn 100 0 0 4 {shapes[0]}"
            test_result = subprocess.run(
                test_cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                check=True
            )
            # print("Test Output:")
            # print(test_result.stdout)
            performance_content = test_result.stdout
    else:
        subprocess.run(
                ["adb", "push" ,cfg.MNN_root / "build_MNN/onnx", "/data/local/tmp/MNN_original"], 
                check=True, 
                stdout=subprocess.DEVNULL
            )
        test_cmd = test_cmd = f"bash {sh_path} ./ModuleBasic.out model.mnn onnx 0 0 100 4"
        test_result = subprocess.run(
            test_cmd, 
            shell=True, 
            # capture_output=True, # adb shell 的输出 不一定走子进程 stdout；Python 的 subprocess 捕获不到
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,#stderr → stdout
            text=True, 
            check=True
        )
        # print("Test Output:")
        # print(test_result.stdout)
        performance_content =  test_result.stdout
    performance = parse_operator_performance(performance_content)

    operator_evaluation_info["compile"] = compile_info
    operator_evaluation_info["correctness"]= correctness
    operator_evaluation_info["performance"] = performance

    print("The whole evaluation of  operator is Done!")
    print(f"operator info:{operator_evaluation_info}")

    return operator_evaluation_info

if __name__ == "__main__":
    # opdic ={
    #     # "ReduceSum":{
    #     #     "op_type":"Atomic",
    #     #     "op_cat":"reduction"
    #     # },
    #     "OneHot":{
    #         "op_type":"Atomic",
    #         "op_cat":"normal"
    #     },
    #     "Range":{
    #         "op_type":"Atomic",
    #         "op_cat":"normal"
    #     }
    # }
    # for name,op_t in opdic.items():
    #     print(name,":", op_t["op_cat"],",",op_t["op_cat"])
    #     cfg = BasicConfig(op_name=name,op_type=op_t["op_type"],op_ctg=op_t["op_cat"])
    #     operator_evaluation_info = gen_and_test_single_op_performance_MNN(cfg)

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--op-name", type=str,default="ReduceSum")
    parser.add_argument("--op-type", type=str,default="Atomic")
    parser.add_argument("--op-ctg", type=str,default="reduction")
    parser.add_argument("--basic-path-config", type=str,default="MobileDeviceKernelBench/scripts/eva_config_MNN.yaml",required=True,help="path of basic config yaml file")

    args = parser.parse_args()
    cfg = OpConfig()
    cfg.update_from_yaml(args.basic_path_config)
    cfg.update_op(args.op_name,args.op_type,args.op_ctg)
    operator_evaluation_info = gen_and_test_single_op_performance_MNN(cfg)

