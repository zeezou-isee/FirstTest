import os
from pathlib import Path
from gen_and_test_single_op_performance import gen_and_test_single_op_performance
from utils.utils import restore_files
from prompt.utils import list_all_txt_or_py,parse_op_info_from_path
from evaluate_MNN_original_op import gen_and_test_single_op_performance_MNN
import json
from scripts.BasicConfig import OpConfig
import argparse
import logging
from datetime import datetime

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basic-path-config", type=str,default="/home/zeezou/MNN/MobileDeviceKernelBench/scripts/eva_config.yaml",help="path of basic config yaml file")
    parser.add_argument("--save-path", type=str,default="/home/zeezou/MNN/MobileDeviceKernelBench/evaluation_results/",help="save results folder")
    parser.add_argument("--LLM-list", nargs="*",default=[])
    args = parser.parse_args()

    # config info
    cfg = OpConfig()
    cfg.update_from_yaml(args.basic_path_config)
    if not os.path.exists(args.save_path):
         os.makedirs(args.save_path,exist_ok=True)

    # log 
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    project_root = getattr(cfg, "project_root", Path.cwd())
    log_dir = project_root/ "eval_logs" / now
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler() 
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Parsed arguments: %s", vars(args))

    # save folder config
    eval_results = {}
    LLM_list = [cfg.cfg_LLM] if not args.LLM_list else args.LLM_list
    model_path_list = list_all_txt_or_py(cfg.op_pymodel_foler_path)
    
    # test per LLM
    for LLM in LLM_list:
        LLM_eval_results = {}
        save_path = Path(args.save_path) / f"eval_{LLM}.json"
        for model_path in model_path_list:
            # update op info and test
            op_info = parse_op_info_from_path(model_path)
            cfg.update_op(
                        op_name=op_info["op_name"],
                        op_type=op_info["op_type"],
                        op_ctg=op_info["op_ctg"],
                        LLM=LLM
                        )

            # log operator info in cfg
            logger.info(
                "Start evaluation | LLM=%s | op_name=%s | op_type=%s | op_ctg=%s",
                LLM, getattr(cfg, "op_name", None), getattr(cfg, "op_type", None),getattr(cfg, "op_ctg", None),
            )

            # save result every op
            result = gen_and_test_single_op_performance(cfg)
            LLM_eval_results[op_info["op_name"]] = result
            with open(save_path, 'w') as f:
                    json.dump(LLM_eval_results, f, indent=4)

            # log result         
            logger.info("Evaluation result: %s", result )
        
        eval_results[LLM] = LLM_eval_results
        logger.info("%s %s %s", "*"*20, LLM, "*"*20)

    # save all results after eval
    save_all_path = Path(args.save_path) / f"eval_all.json"
    with open(save_all_path, 'w') as f:
        json.dump(eval_results, f, indent=4)

        

        