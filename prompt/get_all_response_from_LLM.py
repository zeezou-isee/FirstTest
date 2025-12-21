import os
import time
from pathlib import Path
from prompt.utils import list_all_txt_or_py,read_file,query_llm_openrouter,query_llm_alicloude,parse_op_info_from_path
from prompt.prompt_generator import generate_prompt_with_py_model


def gen_response_from_LMM(op_name,
                          op_type,
                          op_ctg,
                          llm_source:str = "gpt-5",
                          api_source = "openrouter",
                          pymodel_root:Path = None,
                          save_response_root:Path=None):

    assert pymodel_root is not None, "The path of python model folder is needed!"
    op_pymodel_foler_path = pymodel_root / op_type / op_ctg / (op_name+".py")
    prompt = generate_prompt_with_py_model(
                                op_name=op_name,
                                op_ctg=op_ctg,
                                op_type=op_type,
                                py_folder=op_pymodel_foler_path,
                            ) #   generate_prompt_from_model()
    
    match api_source:   #   queryserrver(prompt)
        case "openrouter":
            response = query_llm_openrouter(prompt,llm_source)
        case "aliyun":
            response = query_llm_alicloude(prompt,llm_source)
        case _:
            raise ValueError("None source api surported!")
    print(f"{llm_source} has reponse for op:{op_name}!")
    if save_response_root is not None:
        save_path = save_response_root /llm_source/ op_type / op_ctg / (op_name+".txt")
        with open(save_path, "w") as f:
            f.write()
            print(f"Save reponse to:{save_path}!")
        
    return response

def gen_all_response(
                    pyfolder:Path=None,
                    llm_list:list =None,
                    prompt_mod:str="read",
                    save_path_root:Path=Path("/home/zeezou/MNN/MobileDeviceKernelBench/source/mnn_dataset_test/response_60"),
                    prompt_folder:Path=Path("/home/zeezou/MNN/MobileDeviceKernelBench/source/mnn_dataset_test/prompt_60"),
                    ):
    """
        param:
            prompt_mod:"gen","read",
            llm_list:["google/gemini-2.5-flash",
                        "anthropic/claude-sonnet-4.5",
                        "meta-llama/llama-3.1-405b-instruct",
                        "openai/gpt-5", ]
            pyfolder:/home/zeezou/MNN/MobileDeviceKernelBench/source/mnn_dataset_test/model_60
            prompt_folder:/home/zeezou/MNN/MobileDeviceKernelBench/source/mnn_dataset_test/prompt_60
            save_path_root:/home/zeezou/MNN/MobileDeviceKernelBench/source/mnn_dataset_test/response_60
    """
    match prompt_mod:
        case "gen":
            assert pyfolder is not None,"provide the python model folder"
            path_list = list_all_txt_or_py(pyfolder)
        case "read":
            assert prompt_folder is not None,"provide the prompt model folder"
            path_list = list_all_txt_or_py(prompt_folder)
        case _:
            return None
    save_path_root = save_path_root

    if not save_path_root.exists():
        save_path_root.mkdir(parents=True, exist_ok=True)
    if not llm_list:
        llm_list =  [
                "google/gemini-2.5-flash",
                #   "google/gemini-3-pro-preview",
                "anthropic/claude-sonnet-4.5",
                "meta-llama/llama-3.1-405b-instruct",
                "openai/gpt-5",
                #   "openai/gpt-5.1-codex-max"
                ]
    for model in llm_list:
        model_name = model.split("/")[1] if "/" in model else model
        for path in path_list:
            # parse op info: "op_name","op_ctg","op_type"
            op_info = parse_op_info_from_path(path)
            prompt = read_file(path) if prompt_mod =="read"\
                                    else generate_prompt_with_py_model(
                                            op_name=op_info["op_name"],
                                            op_type =op_info["op_type"],
                                            op_ctg =op_info["op_ctg"],
                                            py_folder=pyfolder,
                                        )
            
            save_path = save_path_root / op_info["op_type"] /op_info["op_ctg"]/(op_info["op_name"]+".txt") 
            if os.path.exists(save_path):
                print(f"Skip existing file: {save_path}")
                continue

            save_path.parent.mkdir(parents=True, exist_ok=True)
            print("--"*20,f"Start query {model} for OP: {op_info["op_type"]}","--"*20)
            # query LLM
            start_time = time.time()
            resp = query_llm_openrouter(prompt,model)
            end_time = time.time()
            print(f"Response: {model} response with time:{end_time-start_time:.3f}s")
            # save response
            save_path.write_text(resp, encoding="utf-8")
            print(f"Saved: {save_path}")
            print("--"*20,f"End query {model} for OP: {op_info["op_type"]}","--"*20)

            time.sleep(5)


if __name__ == "__main__":
    
    prompt_mod:str="read",
    save_path_root:Path=Path("/home/zeezou/MNN/MobileDeviceKernelBench/source/mnn_dataset_test/response_60"),
    prompt_folder:Path=Path("/home/zeezou/MNN/MobileDeviceKernelBench/source/mnn_dataset_test/prompt_60"),

    gen_all_response(
                        prompt_mod = prompt_mod,
                        save_path_root=save_path_root,
                        prompt_folder =prompt_folder
                    )
                                    
    
