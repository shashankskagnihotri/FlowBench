from test import evaluate, load_model
import json
#attack_args = {"attack_epsilon": 8/255}
#evaluate(model_name='raft', pretrained_ckpt='kitti', dataset='kitti-2015', dataset_config_path="./ptlflow/datasets.yml")

sintel_model_names = [
    "raft",
    "pwcnet",
    "gma",
    "rpknet",
    "ccmr",
    "craft",
    "dicl",
    "dip",
    "fastflownet",
    "maskflownet",
    "maskflownet_s",
    "flow1d",
    "flowformer",
    "flowformer++",
    "gmflow",
    "hd3",
    "irr_pwc",
    "liteflownet",
    "liteflownet2",
    "liteflownet3",
    "llaflow",
    "matchflow",
    "ms_raft+",
    "rapidflow",
    "scopeflow",
    "skflow",
    "starflow",
    "videoflow_bof",
]

kitti_model_names = [
    "csflow",
    "gmflownet",
]  

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

parameter_dict = {"experiment" : []}
for model_name in kitti_model_names:
    model = load_model(model_name, "kitti-2015")
    if model:
        params_count = count_parameters(model)
        parameter_dict["experiment"].append({"model": model_name, "model_parameters": params_count})


with open("model_params.json", "w") as json_file:
    json.dump(parameter_dict, json_file, indent=4)
