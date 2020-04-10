# AUthor: duguiming
# Description: 基础的配置项目
# Date: 2020-03-31
import os

base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class BaseConfig(object):
    data_path = "./data/"
    train_path = os.path.join(data_path, 'train_data')
    val_path = os.path.join(data_path, "test_data")
    test_path = os.path.join(data_path, "val_data")

    output_path = "./results_save/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    summary_path = os.path.join(output_path, "summaries")
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    model_dir = os.path.join(output_path, "checkpoints/")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    ckpt_prefix = os.path.join(model_dir, "model")
    model_path = ckpt_prefix

    result_path = os.path.join(output_path, "results")
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    log_path = os.path.join(result_path, "log.txt")
