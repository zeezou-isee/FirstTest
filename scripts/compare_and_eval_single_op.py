import sys
sys.path.append("./MobileDeviceKernelBench")
import argparse
from scripts.BasicConfig import OpConfig
from scripts.gen_and_test_single_op_performance import gen_and_test_single_op_performance
from scripts.evaluate_MNN_original_op import gen_and_test_single_op_performance_MNN

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--op-name", type=str,default="ReduceSum")
    parser.add_argument("--op-type", type=str,default="Atomic")
    parser.add_argument("--op-ctg", type=str,default="reduction")
    parser.add_argument("--basic-path-config", type=str,default="MobileDeviceKernelBench/scripts/eva_config.yaml",help="path of basic config yaml file")
    parser.add_argument("--basic-path-config-MNN", type=str,default="MobileDeviceKernelBench/scripts/eva_config.yaml",help="path of basic config yaml file")

    args = parser.parse_args()
    cfg = OpConfig()
    cfg.update_from_yaml(args.basic_path_config_MNN)
    cfg.update_op(args.op_name,args.op_type,args.op_ctg)   
    op_performance = gen_and_test_single_op_performance(cfg)

    cfg_MNN = OpConfig()
    cfg_MNN.update_from_yaml(args.basic_path_config)
    cfg_MNN.update_op(args.op_name,args.op_type,args.op_ctg)  
    op_performance_MNN = gen_and_test_single_op_performance_MNN(cfg_MNN)

    print(f"{cfg.cfg_LLM}: {cfg.op_name}:{op_performance}")
    print(f"MNN: {cfg.op_name}:{op_performance_MNN}")

