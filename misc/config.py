import json
import os
from easydict import EasyDict as edict


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    config = edict(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    paths = json_file.split('/')[1:-1]
    summary_dir = ["./experiments"] + paths + [config.exp_name, "summary/"]
    ckpt_dir = ["./experiments"] + paths + [config.exp_name, "checkpoint/"]
    # print('Summary dir is', summary_dir)
    # print('Checkpoint dir is', ckpt_dir)
    config.summary_dir = os.path.join(*summary_dir)
    config.checkpoint_dir = os.path.join(*ckpt_dir)
    return config
