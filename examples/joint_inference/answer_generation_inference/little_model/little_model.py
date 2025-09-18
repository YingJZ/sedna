import os
import time
import logging
from sedna.common.config import Context
from sedna.common.file_ops import FileOps
from sedna.core.joint_inference import JointInference
from interface import Estimator

LOG = logging.getLogger(__name__)

input_path = Context.get_parameters('input_text')
all_output_path = Context.get_parameters('all_examples_inference_output')
hard_example_edge_output_path = Context.get_parameters('hard_example_edge_inference_output')
hard_example_cloud_output_path = Context.get_parameters('hard_example_cloud_inference_output')

FileOps.clean_folder([
    all_output_path,
    hard_example_cloud_output_path,
    hard_example_edge_output_path
], clean=False)

def get_input_texts(input_path):
    input_texts = []
    if os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            file_path = os.path.join(input_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as f:
                    for line in f:
                        input_texts.append(line.strip())
    else:
        with open(input_path, 'r') as f:
            for line in f:
                input_texts.append(line.strip())
    return input_texts

def output_deal(final_result, is_hard_example, cloud_result, edge_result, filename, input_text):
    with open(os.path.join(all_output_path, f"{filename}.txt"), 'w') as f:
        f.write(f"Input: {input_text}\nOutput: {final_result}\n")

    if not is_hard_example:
        return
    if cloud_result is not None:
        with open(os.path.join(hard_example_cloud_output_path, f"{filename}.txt"), 'w') as f:
            f.write(f"Input: {input_text}\nCloud Output: {cloud_result}\n")
    if edge_result is not None:
        with open(os.path.join(hard_example_edge_output_path, f"{filename}.txt"), 'w') as f:
            f.write(f"Input: {input_text}\nEdge Output: {edge_result}\n")

def main():
    hard_example_mining = JointInference.get_hem_algorithm_from_config()
    inference_instance = JointInference(
        estimator=Estimator,
        hard_example_mining=hard_example_mining
    )
    
    input_texts = get_input_texts(input_path)
    for idx, input_text in enumerate(input_texts):
        is_hard_example, final_result, edge_result, cloud_result = (
            inference_instance.inference([input_text])
        )
        output_deal(
            final_result,
            is_hard_example,
            cloud_result,
            edge_result,
            str(idx),
            input_text
        )

if __name__ == '__main__':
    main() 