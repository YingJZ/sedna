import os
import time
import logging
from sedna.common.config import Context
from sedna.common.file_ops import FileOps
from sedna.core.joint_inference import JointInference
from interface import Estimator

LOG = logging.getLogger(__name__)

all_output_path = Context.get_parameters('all_examples_inference_output')
hard_example_edge_output_path = Context.get_parameters('hard_example_edge_inference_output')
hard_example_cloud_output_path = Context.get_parameters('hard_example_cloud_inference_output')

FileOps.clean_folder([
    all_output_path,
    hard_example_cloud_output_path,
    hard_example_edge_output_path
], clean=False)

def output_deal(final_result, is_hard_example, cloud_result, edge_result, nframe, input_text):
    # 保存所有推理结果
    with open(os.path.join(all_output_path, f"{nframe}.txt"), 'w') as f:
        f.write(f"Input: {input_text}\nOutput: {final_result}\n")
    # 保存 hard example
    if not is_hard_example:
        return
    if cloud_result is not None:
        with open(os.path.join(hard_example_cloud_output_path, f"{nframe}.txt"), 'w') as f:
            f.write(f"Input: {input_text}\nCloud Output: {cloud_result}\n")
    if edge_result is not None:
        with open(os.path.join(hard_example_edge_output_path, f"{nframe}.txt"), 'w') as f:
            f.write(f"Input: {input_text}\nEdge Output: {edge_result}\n")

def main():
    # hard_example_mining = JointInference.get_hem_algorithm_from_config(
    #     threshold_score=0.7
    # )
    hard_example_mining = {"method": "BertRouter"}
    inference_instance = JointInference(
        estimator=Estimator,
        hard_example_mining=hard_example_mining
    )
    LOG.warning(f"type(inference_instance.estimator): {type(inference_instance.estimator)}")
    
    input_text = ["Hello, I am"]
    while True:
        is_hard_example, final_result, edge_result, cloud_result = (
            # inference_instance.inference(input_text, mining_mode="mining-then-inference")   # 注意：实际不是这样实现的！！
        )
        print(f"is_hard_example: {is_hard_example}")
        print(f"len(final_result): {len(final_result)}")
        print(f"final_result: {final_result}") 
        # output_deal(
        #     final_result,
        #     is_hard_example,
        #     cloud_result,
        #     edge_result,
        #     nframe,
        #     input_text
        # )
        # 只推理一次，如需循环推理可去掉 break
        break

if __name__ == '__main__':
    main() 