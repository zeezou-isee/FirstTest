import os
import sys
sys.path.append("./MobileDeviceKernelBench")
from pathlib import Path
from prompt.utils import list_all_txt_or_py,parse_op_info_from_path
from scripts.gen_and_test_single_op_performance import gen_and_test_single_op_performance
from scripts.evaluate_MNN_original_op import gen_and_test_single_op_performance_MNN
import json
from scripts.BasicConfig import OpConfig
import argparse
import logging
from datetime import datetime

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basic-path-config", type=str,default="/home/zeezou/MNN/MobileDeviceKernelBench/scripts/eva_config.yaml"
                        ,help="path to basic config yaml file")
    parser.add_argument("--basic-path-config-MNN", type=str,default="/home/zeezou/MNN/MobileDeviceKernelBench/scripts/eva_config_MNN.yaml"
                        ,help="path to MNN config yaml file")
    parser.add_argument("--save-path", type=str,default="/home/zeezou/MNN/MobileDeviceKernelBench/evaluation_results/",
                        help="save results folder")
    parser.add_argument("--LLM-list", nargs="*",default=[])
    parser.add_argument("--save-success", type=bool,default=False)
    args = parser.parse_args()
    
    # config info
    cfg = OpConfig()
    cfg.update_from_yaml(args.basic_path_config)
    cfg_MNN = OpConfig()
    cfg_MNN.update_from_yaml(args.basic_path_config_MNN)

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
    eval_ompared_results = {}
    LLM_list = [cfg.cfg_LLM] if not args.LLM_list else args.LLM_list
    model_path_list = list_all_txt_or_py(cfg.op_pymodel_foler_path)
    
    # test per LLM
    for LLM in LLM_list:
        LLM_eval_compared_results = {}
        save_path = Path(args.save_path) / f"eval_compare_{LLM}&MNN.json"
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
            # log result
            logger.info("Evaluation result: %s", result )

            if args.save_success:
                if (isinstance(result["performance"], dict) and result["performance"]["avg"] is not None):
                    cfg_MNN.update_op(
                            op_name=op_info["op_name"],
                            op_type=op_info["op_type"],
                            op_ctg=op_info["op_ctg"],
                            LLM=LLM
                            )
                    logger.info(
                        "Start evaluation | MNN | op_name=%s | op_type=%s | op_ctg=%s",
                        getattr(cfg, "op_name", None), getattr(cfg, "op_type", None),getattr(cfg, "op_ctg", None),
                    )
                    MNN_result = gen_and_test_single_op_performance_MNN(cfg_MNN)
                    # log result
                    logger.info("Evaluation result: %s", MNN_result )

                    LLM_eval_compared_results[op_info["op_name"]] = {
                        LLM:result,
                        "MNN":MNN_result,
                    }
                    with open(save_path, 'w') as f:
                            json.dump(LLM_eval_compared_results, f, indent=4)
            else:
                cfg_MNN.update_op(
                        op_name=op_info["op_name"],
                        op_type=op_info["op_type"],
                        op_ctg=op_info["op_ctg"],
                        LLM=LLM
                        )
                logger.info(
                    "Start evaluation | MNN | op_name=%s | op_type=%s | op_ctg=%s",
                    getattr(cfg, "op_name", None), getattr(cfg, "op_type", None),getattr(cfg, "op_ctg", None),
                )
                MNN_result = gen_and_test_single_op_performance_MNN(cfg_MNN)
                # log result
                logger.info("Evaluation result: %s", MNN_result )

                LLM_eval_compared_results[op_info["op_name"]] = {
                        LLM:result,
                        "MNN":MNN_result,
                }
                with open(save_path, 'w') as f:
                        json.dump(LLM_eval_compared_results, f, indent=4)

        eval_ompared_results[LLM] = LLM_eval_compared_results
        logger.info("%s %s %s", "*"*20, LLM, "*"*20)

    # save all results after eval
    if not eval_ompared_results:
        save_all_path = Path(args.save_path) / f"eval_all.json"
        with open(save_all_path, 'w') as f:
            json.dump(eval_ompared_results, f, indent=4)

    print("Evaluation Done!")

        

        