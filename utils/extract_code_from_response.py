import os
import re
from typing import Dict, List
from pathlib import Path
import tqdm


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
import re
from typing import Dict, List, Tuple

def extract_code_qwen3(text: str) -> Dict[str, str]:
    """
    从类似问题描述的 txt 文本中提取多个文件的代码。

    支持两种写法：
    1) 文件名后面紧跟 ```cpp ... ``` 代码块
    2) 文件名后面直接是源码文本（无 ```）

    返回：
      dict: { "CPUArgMax.hpp": "<code...>", "CPUArgMax.cpp": "<code...>", ... }
    """

    # 你可以按需扩展后缀
    exts = r"(?:h|hpp|c|cc|cpp|mm|m|cu|py|java|kt|js|ts|go|rs|cs|swift|proto|txt)"
    filename_re = re.compile(
        rf"""(?mx)
        ^\s*
        (?:\d+\s*[、.．]\s*)?              # 可选：1、 或 1. 等
        (?P<name>[A-Za-z0-9_./\\-]+\.(?:{exts}))  # 文件名
        \s*$
        """
    )

    # 找到所有“文件名行”的位置
    matches: List[re.Match] = list(filename_re.finditer(text))
    if not matches:
        return {}

    # 切分成 (filename, chunk_text)
    chunks: List[Tuple[str, str]] = []
    for idx, m in enumerate(matches):
        name = m.group("name").strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chunk = text[start:end]
        chunks.append((name, chunk))

    def _extract_code_from_chunk(chunk: str) -> str:
        # 优先提取 fenced code blocks（```...```），可能有多个，全部拼接
        fenced = re.findall(r"```[^\n]*\n(.*?)```", chunk, flags=re.S)
        if fenced:
            code = "\n\n".join(s.rstrip() for s in fenced).strip("\n")
            return code

        # 没有 fenced code block，则把整段当代码
        # 去掉段首空行，但保留代码内部格式
        code = chunk.strip("\n")

        # 可选：如果段首有类似“：”或多余空白，可做轻量清理
        code = re.sub(r"^\s*\n+", "", code)
        return code.rstrip()

    result: Dict[str, str] = {}
    for name, chunk in chunks:
        code = _extract_code_from_chunk(chunk)
        # 若同名文件重复出现，则追加（也可以改成覆盖）
        if name in result and code:
            result[name] = (result[name].rstrip() + "\n\n" + code.lstrip()).strip("\n")
        else:
            result[name] = code
    if not result:
        print("Use anothe method.")
        result = extract_GPT_llama_claude(text)
    return result

def extract_code_qwen3_0(txt: str) -> Dict[str, str]:
    """
    Extract code modules from raw text.
    Supports:
      - filename + ```cpp code block
      - filename + raw code
      - single file without filename

    Returns:
        Dict[str, str]: {filename: code}
    """

    filename_pattern = re.compile(r'^[\w\-]+\.(cpp|hpp|h|cc|cxx)$')
    fence_pattern = re.compile(r'^```')

    lines = txt.splitlines()
    modules = {}

    current_file = None
    buffer = []
    in_fence = False

    for line in lines:
        stripped = line.strip()

        # 1. 文件名检测
        if not in_fence and filename_pattern.match(stripped):
            # 保存前一个
            if current_file and buffer:
                modules[current_file] = "\n".join(buffer).strip()
            current_file = stripped
            buffer = []
            continue

        # 2. code fence 开始 / 结束
        if fence_pattern.match(stripped):
            if in_fence:
                # fence 结束，保存
                if current_file:
                    modules[current_file] = "\n".join(buffer).strip()
                buffer = []
                in_fence = False
            else:
                in_fence = True
            continue

        # 3. 收集代码
        if current_file:
            if in_fence or not fence_pattern.match(stripped):
                buffer.append(line)

    # 4. 收尾
    if current_file and buffer:
        modules[current_file] = "\n".join(buffer).strip()

    # 5. 兜底：只有一个代码块，没有文件名
    if not modules:
        code = []
        in_fence = False
        for line in lines:
            if fence_pattern.match(line.strip()):
                in_fence = not in_fence
                continue
            if in_fence:
                code.append(line)
        if code:
            modules["unknown.cpp"] = "\n".join(code).strip()
        elif txt.strip():
            modules["unknown.cpp"] = txt.strip()

    return modules
 
def extract_GPT_llama_claude(prompt_resp: str) -> Dict[str, str]:
    """
    Extract C++ module code blocks (.cpp / .hpp) from GPT response.
    
    Args:
        prompt_resp (str): GPT full text response.
    Returns:
        Dict[str, str]: {"CPUGridSample.hpp": "...", "CPUGridSample.cpp": "..."}
    """

    # 找所有 ```...``` 的代码块（最鲁棒，避免误匹配）
    code_blocks = re.findall(r"```(?:cpp|c\+\+|hpp|.*?)\s*(.*?)```", 
                             prompt_resp, 
                             flags=re.DOTALL | re.IGNORECASE)

    results = {}

    for block in code_blocks:
        lines = block.strip().split("\n")
        filename = None

        # 尝试从第一段注释的文件路径中提取名字
        # 例如: // CPUGridSample.hpp
        for line in lines[:5]:  # 只在前5行查找
            m = re.search(r"//\s*([A-Za-z0-9_]+\.(?:hpp|cpp))", line)
            if m:
                filename = m.group(1).strip()
                break

        # 如果找不到文件名，则跳过该代码块
        if filename is None:
            continue

        # 保存文件
        results[filename] = block.strip()
    if not results:
        results = extract_deepseek(prompt_resp)

    return results
import re
from typing import Dict, List

def extract_deepseek(txt: str) -> Dict[str, str]:
    """
    Extract code blocks in the pattern:
        filename(.cpp/.hpp):
        ```cpp
        code
        ```

    Returns:
        Dict[str, str]: {filename: code}
    """

    filename_pattern = re.compile(
        r'^\s*([\w\-]+\.(?:cpp|hpp|h|cc|cxx))\s*:?\s*$'
    )
    fence_start_pattern = re.compile(r'^\s*```')
    fence_end_pattern = re.compile(r'^\s*```')

    lines = txt.splitlines()
    modules: Dict[str, str] = {}

    current_file = None
    collecting = False
    buffer = []

    for line in lines:
        # 1. 文件名行
        m = filename_pattern.match(line)
        if m and not collecting:
            current_file = m.group(1)
            continue

        # 2. fence 开始
        if current_file and fence_start_pattern.match(line) and not collecting:
            collecting = True
            buffer = []
            continue

        # 3. fence 结束
        if collecting and fence_end_pattern.match(line):
            modules[current_file] = "\n".join(buffer).strip()
            collecting = False
            current_file = None
            buffer = []
            continue

        # 4. 收集代码
        if collecting:
            buffer.append(line)
    if not modules:
        modules = extract_Gemini(txt)
    return modules
def extract_Gemini(prompt_resp: str) -> str:
    """
    Extract cpp/hpp modules from Gemini response.
    Module format must be:
        FileName.hpp (or .cpp)
        ```
        code...
        ```
    Returns concatenated modules as a single string.
    """

    # 正则匹配格式：
    # 文件名（单独一行） + 紧随其后的 ```code``` 模块
    pattern = re.compile(
        r"^\s*([A-Za-z0-9_]+\.(?:hpp|cpp))\s*?\n```(?:[A-Za-z0-9_+-]*)?\s*(.*?)```",
        re.DOTALL | re.MULTILINE
    )

    matches = pattern.findall(prompt_resp)
    if not matches:
        print("Try another method")
        code_blocks = re.findall(r"```(?:cpp|c\+\+|hpp|.*?)\s*(.*?)```", 
                             prompt_resp, 
                             flags=re.DOTALL | re.IGNORECASE)
        results = {}

        for block in code_blocks:
            lines = block.strip().split("\n")
            filename = None

            # 尝试从第一段注释的文件路径中提取名字
            # 例如: // CPUGridSample.hpp
            for line in lines[:5]:  # 只在前5行查找
                m = re.search(r"//\s*([A-Za-z0-9_]+\.(?:hpp|cpp))", line)
                if m:
                    filename = m.group(1).strip()
                    break
            
            # 如果找不到文件名，则跳过该代码块
            if filename is None:
                continue

            # 保存文件
            results[filename] = block.strip()

        return results
    else:
        result_parts = {}

        for filename, code in matches:
            code = code.rstrip()  # 清理多余换行
            result_parts[filename] = code

        return result_parts
def remove_think_block(txt: str) -> str:
    """
    Remove content between <think> and </think>, including the tags.
    Works for multiline text.
    """
    pattern = re.compile(r"<think>.*?</think>", re.DOTALL)
    return re.sub(pattern, "", txt).strip()

def extract_code_from_response(prompt_path_or_response: str, source_model: str,op_type:str= "Atomic",op_catogry:str="normal"):
    """
        prompt_path: absolute path of response txt file
        source_model: gemini-2.5-flash / gpt-5 / claude-sonnet-4.5 / llama-3.1-405b-instruct

        return:
            dict:{
            "filename.cpp": "code content", eg: "CPUGridSample.cpp": "...."
            ...
            }
    """
    supported_models = {"gemini-2.5-flash", "gpt-5", "claude-sonnet-4.5", "llama-3.1-405b-instruct","qwen3-235b-a22b-thinking-2507","deepseek-r1-0528"}
    # assert source_model in supported_models, (
    #     "supported models: gemini-2.5-flash, gpt-5, claude-sonnet-4.5, llama-3.1-405b-instruct, qwen3-235b-a22b-thinking-2507,deepseek-r1-0528"
    # )
    if prompt_path_or_response.endswith(".txt"):
        with open(prompt_path_or_response, "r", encoding="utf-8") as f:
            prompt_resp = f.read()
            path_s = str(prompt_path_or_response)
    else:
        prompt_resp = prompt_path_or_response
    if source_model == "qwen3-8B":
        prompt_resp = remove_think_block(prompt_resp)
        code_dic = extract_deepseek(prompt_resp)

    if source_model=="gemini-2.5-flash" :
        code_dic = extract_Gemini(prompt_resp)
    elif source_model == "deepseek-r1-0528":
        code_dic = extract_deepseek(prompt_resp)
    elif source_model == "qwen3-235b-a22b-thinking-2507":
        code_dic = extract_code_qwen3(prompt_resp)
    else:
        code_dic = extract_GPT_llama_claude(prompt_resp)

    match op_type:
        case "Atomic":
            match op_catogry:
                case "binary":
                    allowed = {"CPUBinary.cpp"}
                case "unary":
                    allowed = {"CPUUnary.cpp"}
                case "reduction":
                    allowed = {"CPUReduction.cpp"}
                case "normal":
                    allowed = None
                    return code_dic if len(code_dic) == 2 else None
                case "convolution":
                    allowed = allowed = {
                            "DenseConvolutionTiledExecutor.cpp",
                            "ConvolutionGroup.cpp",
                            "Convolution1x1Strassen.cpp",
                            "ConvolutionPackWinograd.cpp",
                        }

            return {k: v for k, v in code_dic.items() if k in allowed}
        case "Combiner":
            return code_dic
        case "Geometry":
            return code_dic
        case _:
            raise ValueError(f"Unknown op type in path: {prompt_path_or_response}")

if __name__ == "__main__":
    # resp_path = Path("/Users/zeezou/python/project/MNN-master/dataset_mnn_test/response_60/gemini-2.5-flash/Combiner/normal/LogSoftmax.txt")
    # with open(resp_path, "r", encoding="utf-8") as f:
    #     resp = f.read()
    # code = extract_Gemini(resp)
    # print(code)
    # exit()
    # res_path = "/home/zeezou/MNN/MobileDeviceKernelBench/source/mnn_dataset_test/response_60/gemini-2.5-flash/Geometry/normal/DepthToSpace.txt"
    # source_model = res_path.split("/")[-4]
    # op_type = res_path.split("/")[-3]
    # op_ctg = res_path.split("/")[-2]
    # code = extract_code_from_response(str(res_path), source_model,op_type=op_type,op_catogry=op_ctg)
    # # print(code)
    # for filename, content in code.items():
    #     print(f"  File: {filename}\n")
    # exit()
    # response_path = Path("/home/zeezou/MNN/MobileDeviceKernelBench/source/mnn_dataset_test/response_60") / "qwen3-235b-a22b-thinking-2507"
    # response_path = Path("/home/zeezou/MNN/MobileDeviceKernelBench/source/mnn_dataset_test/response_60") / "deepseek-r1-0528"
    # response_path = Path("/home/zeezou/MNN/MobileDeviceKernelBench/source/mnn_dataset_test/response_60") / "claude-sonnet-4.5"
    # response_path = Path("/home/zeezou/MNN/MobileDeviceKernelBench/source/mnn_dataset_test/response_60") / "gemini-2.5-flash"
    # response_path = Path("/home/zeezou/MNN/MobileDeviceKernelBench/source/mnn_dataset_test/response_60") / "llama-3.1-405b-instruct"
    # response_path = Path("/home/zeezou/MNN/MobileDeviceKernelBench/source/mnn_dataset_test/response_60") / "claude-sonnet-4.5"
    response_path = Path("/home/zeezou/MNN/MobileDeviceKernelBench/source/mnn_dataset_test/response_60") / "qwen3-8B"
    res_list = list_all_txt_or_py(response_path)

    # test return code block
    for res_path in res_list:
        # print(f"Processing file: {res_path}")
        source_model = res_path.split("/")[-4]
        op_type = res_path.split("/")[-3]
        op_ctg = res_path.split("/")[-2]
        code = extract_code_from_response(str(res_path), source_model,op_type=op_type,op_catogry=op_ctg)
        if not code:
            print("Fail path:",res_path)
            continue
        # for filename, content in code.items():
        #     print(f"  File: {filename}\n")
            # print(content)
    exit()
    # save in folder:extracted_code_60
    save_code_path = ""
    pbar = tqdm.tqdm(total=len(res_list), desc="Extracting code from responses...")
    for res_path in res_list:
        print(f"Processing file: {res_path}")

        save_code_path = Path(str(res_path).replace("response_60","extracted_code_60"))
        
        save_code_path.parent.mkdir(parents=True, exist_ok=True)

        with open(res_path, "r", encoding="utf-8") as f:
            resp = f.read()

        if "gemini" in str(res_path):
            codefiles = extract_Gemini(resp)
        else:
            codefiles = extract_GPT_llama_claude(resp)

        for filename, code in codefiles.items():
            save_path = save_code_path.parent / filename
            if os.path.exists(save_path):
                print(f"Skip existing file: {save_path}")
                continue

            with open(save_path, "w", encoding="utf-8") as codef:
                codef.write(code)

            print(f"  Saved extracted code to: {save_path}")

            pbar.update(1)

    pbar.close()
        
    # testfile = "/Users/zeezou/python/project/MNN-master/dataset_mnn_test/response_60/claude-sonnet-4.5/Atomic/normal/ArgMax.txt"
    # resp = ""
    # with open(testfile, "r", encoding="utf-8") as f:
    #     resp = f.read()

    # codefiles = extract_GPT_llama_claude(resp)
    # for filename, code in codefiles.items():
    #     print(f"File: {filename}\n")
    #     print(code)