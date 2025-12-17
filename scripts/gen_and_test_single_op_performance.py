import os
from pathlib import Path
import sys
sys.path.append("./MobileDeviceKernelBench")
from utils.utils import (
                            restore_files,
                            register_op_mnn,
                            compile_project,
                            compile_project_withbar,
                            test_correctness,
                            test_performance,
                            test_performance_withbar,
                            parse_operator_performance,
                        )
from scripts.BasicConfig import OpConfig
from utils.extract_code_from_response import extract_code_from_response
from prompt.prompt_generator import generate_prompt_with_py_model,read_file
from prompt.utils import query_llm_openrouter,query_llm_alicloude
import subprocess
import argparse

def gen_and_test_single_op_performance(cfg:OpConfig):
    """
        main test interface here
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
        "register":"",
        "compile":"",
        "correctness": "",
        "performance": "",
    }

    # 1. get LLM response
    # chose evaluation mod: gen or read from response
    match cfg.response_mod:
        case "gen":
            prompt = generate_prompt_with_py_model(
                                        op_name=cfg.op_name,
                                        op_ctg=cfg.op_ctg,
                                        op_type=cfg.op_type,
                                        py_folder=cfg.op_pymodel_foler_path,
                                    ) #   generate_prompt_from_model()
            match cfg.api_source:   #   queryserrver(prompt)
                case "openrouter":
                    response = query_llm_openrouter(prompt,cfg.cfg_LLM)
                case "aliyun":
                    response = query_llm_alicloude(prompt,cfg.cfg_LLM)
                case _:
                    raise ValueError("None source api surported!")
        case "read":
            # assert cfg.response_file is not None, "Error! A txt file of LLM response is needed!"
            if not os.path.exists(cfg.response_file):
                return operator_evaluation_info
            with open(cfg.response_file, "r") as f:
                response = f.read() 
        case _:
            return operator_evaluation_info

    # 2. extract code from LLM response
    print("path:",cfg.response_file)
    code_block = extract_code_from_response(response,source_model = cfg.cfg_LLM,op_type=cfg.op_type,op_catogry=cfg.op_ctg)
    # print(cfg.cfg_LLM,cfg.op_type)
    # print(code_block)
    # exit()
    if not code_block:
        print("operator find error!")
        return operator_evaluation_info
    
    # 3. register op to MNN
    register = register_op_mnn(
                            code_block,
                            target_folder=cfg.register_op_folder,
                            backup_dir=cfg.backup_folder
                        )
    if not register:
        print("Register error! Check MNN project!")
        return operator_evaluation_info
    register_info ="Register success!"

    # 4.compile MNN with new op
    print("Start Compile MNN with new operator")
    compiled = compile_project_withbar(op_name=cfg.op_name, 
                               llm=cfg.cfg_LLM, 
                               op_type = cfg.op_type,
                               op_ctg = cfg.op_ctg,
                               target_folder=cfg.register_op_folder,
                               log_root=cfg.project_root,
                               MNN_root = cfg.MNN_root,
                               backup_dir=cfg.backup_folder
                               )
    # compiled = True
    if compiled:
        # print("MNN compile with new operator success!")
        compile_info = "pass"

    # 5. test correctness of new op
    # print("Start Test Correctness new operator")
    correct, num_inputs, shapes = test_correctness(compile=compiled,
                                                   MNN_root=cfg.MNN_root,
                                                    onnx_model=cfg.op_onnx_path,
                                                    mnn_model_path=cfg.op_mnn_path
                                                    )
   
    # 6. test performance of new op    
    #   6.1 push test model to mobile phone
    if correct:
        # print("New operator correctness varification success!")
        correctness = "Model output is correct!"

        print("adb push model to phone")
        subprocess.run(
                ["adb","push",str(cfg.op_mnn_path),"/data/local/tmp/MNN/model.mnn"],
                check=True, 
                stdout=subprocess.DEVNULL
            )
    #   6.2 test performance
    # print("Test the performance of new operator")
    performance_content = test_performance_withbar(num_inputs=num_inputs,
                                           shapes=shapes,
                                           correctness=correct,
                                           MNN_root=cfg.MNN_root,
                                           project_root=cfg.project_root
                                           )
    performance = parse_operator_performance(performance_content)

    # 7. restore original MNN op file
    restore_files(backup_dir= cfg.backup_folder,
                    target_folder=cfg.register_op_folder)

    operator_evaluation_info["register"] = register_info
    operator_evaluation_info["compile"] = compile_info
    operator_evaluation_info["correctness"]= correctness
    operator_evaluation_info["performance"] = performance

    print("The whole evaluation of new operator is Done!")
    print(f"operator info:{operator_evaluation_info}")

    return operator_evaluation_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--op-name", type=str,default="Gather")
    parser.add_argument("--op-type", type=str,default="Geometry")
    parser.add_argument("--op-ctg", type=str,default="normal")
    parser.add_argument("--basic-path-config", type=str,default="MobileDeviceKernelBench/scripts/eva_config.yaml",help="path of basic config yaml file")

    args = parser.parse_args()
    cfg = OpConfig()
    cfg.update_from_yaml(args.basic_path_config)
    cfg.update_op(args.op_name,args.op_type,args.op_ctg)    
    operator_evaluation_info = gen_and_test_single_op_performance(cfg)

