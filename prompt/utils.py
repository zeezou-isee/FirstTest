import os
from openai import OpenAI
from pathlib import Path
import time

OPENROUTER_KEY=os.environ.get("OPENROUTER_KEY") 
            #"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"
ALIYUN_KEY=os.environ.get("ALIYUN_KEY")
            #"sk-90fdd16b2ff74530a1ee9e8ca168fc60"

def parse_op_info_from_path(path_of_operator: str|Path):
    op_info = {}
    p_l = str(path_of_operator).split("/") 
    op_name = p_l[-1].split(".")[0]
    op_ctg = p_l[-2]
    op_type = p_l[-3]

    op_info ={
        "op_name":op_name,
        "op_type":op_type,
        "op_ctg":op_ctg
    }
    return op_info

def read_file(file_path) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""
    
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def list_all_txt_or_py(root_dir):
    paths = []

    def recurse(path):
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)

            if os.path.isdir(full_path):
                recurse(full_path)
            elif entry.lower().endswith(".txt") or entry.lower().endswith(".py"):
                paths.append(full_path)

    recurse(root_dir)
    return paths


def query_llm_openrouter(prompt:str = None,model:str = ""):
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_KEY,
        )
    completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "", # Optional. Site URL for rankings on openrouter.ai.
                    "X-Title": "", # Optional. Site title for rankings on openrouter.ai.
                },
                model=model,
                messages=[
                    {
                    "role": "user",
                    "content": prompt
                    }
                ]
    )
    content = completion.choices[0].message.content

    with open("Benchmark_pipeline/testfolder/openrouter_response.txt","w",encoding="utf-8") as f:
        f.write(content)
    print(content)
    print("Token usage:", completion.usage)
    return content


def query_llm_alicloude(prompt:str = None, models =None):
    if models is None:
        models =["qwen2.5-coder-32b-instruct",
                 "qwen3-235b-a22b-thinking-2507",
                 "deepseek-r1-0528",
                 "deepseek-v3"]
    client = OpenAI(
        # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
        api_key="sk-90fdd16b2ff74530a1ee9e8ca168fc60",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        # 此处以 deepseek-v3.2-exp 为例，可按需更换模型名称为 deepseek-v3.1、deepseek-v3 或 deepseek-r1
        model=models,
        messages=messages,
        # 通过 extra_body 设置 enable_thinking 开启思考模式，该参数仅对 deepseek-v3.2-exp 和 deepseek-v3.1 有效。deepseek-v3 和 deepseek-r1 设定不会报错
        extra_body={"enable_thinking": True,"Thinking_budget":81920 if "qwen3" in models else 38912},
        stream=False,
        # stream_options={
        #     "include_usage": True
        # },
    )

    reasoning_content = ""  # 完整思考过程
    answer_content = ""  # 完整回复
    choice = completion.choices[0]
    message = choice.message

    # 提取思考过程（如果有）
    if hasattr(message, "reasoning_content") and message.reasoning_content:
        reasoning_content = message.reasoning_content

    # 提取最终回答
    if hasattr(message, "content") and message.content:
        answer_content = message.content

    # 输出
    print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")
    print(reasoning_content)

    print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
    print(answer_content)

    print("\n" + "=" * 20 + "Token 消耗" + "=" * 20 + "\n")
    print(completion.usage)
    return answer_content

def list_all_txt(root_dir):
    paths = []

    def recurse(path):
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)

            if os.path.isdir(full_path):
                recurse(full_path)
            elif entry.lower().endswith(".txt"):
                paths.append(full_path)

    recurse(root_dir)
    return paths



if __name__ == "__main__":
    prompt_path_list = list_all_txt(Path("/home/zeezou/MNN/MobileDeviceKernelBench/source/mnn_dataset_test/prompt_60"))
    save_path_root = Path("/home/zeezou/MNN/MobileDeviceKernelBench/source/mnn_dataset_test/response_60")

    if not save_path_root.exists():
        save_path_root.mkdir(parents=True, exist_ok=True)

    # model_list = models = [
    #         "google/gemini-2.5-flash",
    #         #   "google/gemini-3-pro-preview",
    #           "anthropic/claude-sonnet-4.5",
    #           "meta-llama/llama-3.1-405b-instruct",
    #           "openai/gpt-5",
    #         #   "openai/gpt-5.1-codex-max"
    #           ]
    model_list = [
                # "qwen2.5-coder-32b-instruct",
                 "qwen3-235b-a22b-thinking-2507",
                 "deepseek-r1-0528",
                #  "deepseek-v3"
                 ]
    for model in model_list:
        for path in prompt_path_list:
            # 读取 prompt 文件内容
            # path = "/Users/zeezou/python/project/MNN-master/dataset_mnn_test/prompt_60/Atomic/reduction/ReduceSum.txt"
            prompt = read_file(path)
            model_name = model.split("/")[1] if "/" in model else model
            # 将原路径从 prompt_60 替换成 response_60
            save_path = Path(str(path).replace("prompt_60", f"response_60/{model_name}"))
            if os.path.exists(save_path):
                print(f"Skip existing file: {save_path}")
                continue

            # 创建保存文件所在的目录
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # 调用 LLM
            # resp = query_llm_openrouter(prompt,model)
            resp = query_llm_alicloude(prompt,model)
            # exit()
            # resp = ""

            # 保存结果
            save_path.write_text(resp, encoding="utf-8")

            print(f"Saved: {save_path}")
            time.sleep(5)
        
        # print(save_path_root / f"{path.stem}_response.txt")
        # with open(save_path_root / f"{path.stem}_response.txt", "w", encoding="utf-8") as f:
        #     f.write(resp)
    # query_llm_alicloude()
    # prompt = "who are you?"
    # resp = query_qwen3_8B(prompt)
    # print(resp)