import os
from pathlib import Path
from gen_and_test_single_op_performance import (
    gen_and_test_single_op_performance,
    BasicConfig
)
from utils.utils import restore_files
from prompt.utils import list_all_txt_or_py,parse_op_info_from_path
from evaluate_MNN_original_op import gen_and_test_single_op_performance_MNN
import json
def gen_and_test_all_ops(response_folder:Path, llm_list:list=None,):
    all_results = {}
    all_llm_op = {
        "Atomic": [
            "And",
            "Div",
            "Dense_Convolution_2D",
            "Group_Convolution_2D",
            "Strassen_Convolution_2D",
            "Winograd_Convolution_2D",
            "ArgMax",
            "ArgMin",
            "Cast",
            "Deconvolution_1D",
            "Deconvolution_2D__asymmetric_input__asymmetric_kernel",
            "Deconvolution_2D_asymmetric_input_square_kernel___dilated____padded____strided__",
            "DeconvolutionDepthwise_2D_kernel",
            "DeconvolutionDepthwise_2D_stride",
            "GridSample",
            "LayerNorm",
            "MatMul",
            "OneHot",
            "Range",
            "Relu",
            "Softmax",
            "Stft",
            "Where",
            "ReduceMean",
            "ReduceMin",
            "ReduceSum",
            "ATen",
            "Cos"
        ],
        "Combiner": [
            "AveragePool",
            "BatchNormalization",
            "Celu",
            "Clip",
            "Gemm",
            "HardSigmoid",
            "InstanceNormalization",
            "LayerNormalization",
            "LogSoftmax",
            "MaxPool",
            "MeanVarianceNormalization",
            "PRelu",
            "Softmax",
            "Softplus"
        ],
        "Geometry": [
            "And",
            "Div",
            "Dense_Convolution_2D_kernel",
            "Group_Convolution_2D",
            "Strassen_Convolution_2D",
            "Winograd_Convolution_2D",
            "ConvTranspose2D__asymmetric_input__asymmetric_kernel",
            "DepthToSpace",
            "Gather",
            "Layernorm",
            "Reshape",
            "Slice",
            "Tile",
            "TopK",
            "ReduceMax",
            "ReduceMean",
            "Asin",
            "ATan"
        ]
        }
    # claude = gemini= gpt= llama = all_llm_op
    claude = {
            "Atomic": [
                "Group_Convolution_2D", "ArgMax", "ArgMin",
                "LayerNorm", "Range", "Relu", "Softmax",
                "Where", "ReduceMean", "ReduceMin", "ReduceSum"
            ],
            "Combiner": [
                "BatchNormalization", "Celu", "HardSigmoid",
                "LogSoftmax", "MaxPool", "Softmax", "Softplus"
            ],
            "Geometry": [
                "DepthToSpace", "Reshape", "Slice"
            ],

        }
    gemini = {
                "Atomic": [
                    "Range", "Softmax", "Where", "ReduceMin", "ReduceSum"
                ],
                "Combiner": [
                    "LogSoftmax", "Softmax", "Softplus"
                ],
                "Geometry": [
                    "Layernorm"
                ],
            }
    gpt= {
        "Atomic": [
            "OneHot", 
            "ReduceMean", "ReduceMin", "ReduceSum"
        ],
        "Combiner": [
            "Gemm", "HardSigmoid", "InstanceNormalization",
            "MeanVarianceNormalization", "Softplus"
        ],
        "Geometry": [
            "Group_Convolution_2D", "Winograd_Convolution_2D",
            "Reshape", "Slice", "ReduceMax", "ATan","And"
        ],
        }
    
    
    llama = {
            "Atomic": [
                "Range", "ReduceSum",
                "where"
            ],
            "Combiner": [],
            "Geometry": [],
                }
    
    qwen3_235b =all_llm_op

    if llm_list is None:
        llm_list = {
            'gpt-5':gpt,
            # "claude-snnoet-4.5":claude,
            # "gemini-2.5-flash":gemini,
            "llama-3.1-405b-instruct":llama,
            # "qwen3-235b-a22b-thinking-2507":qwen3_235b
        }
    
    for llm,op_dic in llm_list.items():
        resp_folder = response_folder / llm
        response_file_path_list = list_all_txt_or_py(resp_folder)
        
        current_llm_results = {}
        for response_file_path in response_file_path_list:
            
            op_info = parse_op_info_from_path(response_file_path)

            if op_info["op_name"] in op_dic[op_info["op_type"]]:
                # print(response_file_path)
                print(f"test info: {llm}, op: {op_info}")
                cfg = BasicConfig(
                    op_name = op_info["op_name"],
                    op_type = op_info["op_type"],
                    op_ctg = op_info["op_ctg"],
                    llm = llm,
                )
                # try:
                #     result = gen_and_test_single_op_performance(cfg)
                #     operator_evaluation_info = {
                #         "ok": True,
                #         "result": result,
                #         "error": None,
                #     }
                # except Exception as e:
                #     restore_files(cfg.backup_folder,cfg.register_op_folder)
                    
                #     operator_evaluation_info = {
                #         "ok": False,
                #         "result": None,
                #         "error": str(e),
                #     }
                try:
                    result = gen_and_test_single_op_performance(cfg)
                    MNN_result = gen_and_test_single_op_performance_MNN(cfg)
                    operator_evaluation_info = {
                        "ok": True,
                        "LLM_result": result,
                        "MNN_result":MNN_result,
                        "error": None,
                    }
                    current_llm_results[op_info["op_name"]] = operator_evaluation_info
                    
                    # save per op result here
                    results_dir = cfg.project_root / 'evaluation_results'
                    llm_filename = results_dir / f"{llm}_results.json"
                    os.makedirs(results_dir,exist_ok=True)
                    with open(llm_filename, 'w') as f:
                        json.dump(current_llm_results, f, indent=4)
                    print(f"Saved results for {llm} to {llm_filename}")
                except Exception as e:
                    restore_files(cfg.backup_folder,cfg.register_op_folder)
                    print("--"*20,f"op test finish","--"*20)
            else:
                print("--"*20,f"Exist! SKIP","--"*20)
            
        #  Add to the unified results dictionary
        all_results[llm] = current_llm_results
    #  Save the unified json file containing all LLMs
    unified_filename = results_dir / "all_llm_results.json"
    with open(unified_filename, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Saved unified results to {unified_filename}")

if __name__=="__main__":
    response_folder = Path("/home/zeezou/MNN/MobileDeviceKernelBench/source/mnn_dataset_test/response_60")
    gen_and_test_all_ops(response_folder=response_folder)