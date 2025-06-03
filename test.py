"""Validate optical flow estimation performance on standard datasets."""

# =============================================================================
# Copyright 2021 Henrique Morimitsu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import pdb
import yaml
import logging
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

import os

import cv2 as cv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

current_directory = os.path.dirname(os.path.abspath(__file__))
ptl_directory = os.path.join(current_directory, "ptlflow")

sys.path.insert(0, ptl_directory)

from ptlflow import get_model, get_model_reference
from ptlflow.models.base_model.base_model import BaseModel
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter
from ptlflow.utils.utils import (
    add_datasets_to_parser,
    config_logging,
    get_list_of_available_models_list,
    tensor_dict_to_numpy,
)

from attacks.fgsm import fgsm
from attacks.apgd import apgd
from attacks.bim_pgd_cospgd import bim_pgd_cospgd
from attacks.fab import fab
from attacks.pcfa import pcfa
from attacks.tdcc import get_dataset_3DCC
from attacks.common_corruptions import common_corrupt
from attacks.attack_utils.attack_args_parser import AttackArgumentParser
from attacks.attack_utils.attack_args_parser import (
    attack_targeted_string,
    attack_arg_string,
)
import attacks.attack_utils.loss_criterion as losses
from attacks.attack_utils.loss_criterion import LossCriterion
from validate import (
    _get_model_names,
)

# Import cosPGD functions
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from attacks.attack_utils.utils import get_flow_tensors, get_image_tensors


# Default Attack parameters
epsilon = 8 / 255
norm = "inf"
alpha = 0.01
iterations = 3
loss_function = "epe"
targeted = False
batch_size = 1

delta_bound = 0.005


config_logging()

# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
import argparse
def parse_fraction(value):
    try:
        # Check if it's a fraction (contains a '/')
        if '/' in value:
            num, denom = value.split('/')
            return float(num) / float(denom)
        else:
            # Otherwise, convert it to a float directly
            return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid fraction or float value: {value}")


def _init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["all", "select"] + get_list_of_available_models_list(),
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="none",
        nargs="*",
        choices=[
            "fgsm",
            "bim",
            "pgd",
            "cospgd",
            "ffgsm",
            "apgd",
            "fab",
            "pcfa",
            "3dcc",
            "common_corruptions",
            "none",
        ],
        help="Name of the attack to use.",
    )
    parser.add_argument(
        "--attack_optim_target",
        type=str,
        default="ground_truth",
        nargs="*",
        choices=["ground_truth", "initial_flow"],
        help="Set optimization target for adversarial attacks.",
    )
    parser.add_argument(
        "--cc_name",
        type=str,
        default="gaussian_noise",
        nargs="*",
        choices=[
            "gaussian_noise",
            "shot_noise",
            "impulse_noise",
            "defocus_blur",
            "glass_blur",
            "motion_blur",
            "zoom_blur",
            "snow",
            "frost",
            "fog",
            "brightness",
            "contrast",
            "elastic_transform",
            "pixelate",
            "jpeg_compression",
        ],
        help="Name of the common corruption to use on the input images.",
    )
    parser.add_argument(
        "--cc_severity",
        type=int,
        default=1,
        nargs="*",
        choices=[1, 2, 3, 4, 5],
        help="Severity of the common corruption to use on the input images.",
    )
    parser.add_argument(
        "--attack_norm",
        type=str,
        default=norm,
        nargs="*",
        choices=["two", "inf"],
        help="Set norm to use for adversarial attack.",
    )
    parser.add_argument(
        "--attack_epsilon",
        type=parse_fraction,
        default=epsilon,
        nargs="*",
        help="Set epsilon to use for adversarial attack.",
    )
    parser.add_argument(
        "--pcfa_boxconstraint",
        default="change_of_variables",
        nargs="*",
        choices=["clipping", "change_of_variables"],
        help="the way to enfoce the box constraint on the distortion. Options: 'clipping', 'change_of_variables'.",
    )
    parser.add_argument(
        "--apgd_rho",
        default=0.75,
        nargs="*",
        type=float,
        help="parameter for step-size update (Default: 0.75)",
    )
    parser.add_argument(
        "--apgd_n_restarts",
        default=1,
        nargs="*",
        type=int,
        help="number of random restarts. (Default: 1)",
    )
    parser.add_argument(
        "--apgd_eot_iter",
        default=1,
        nargs="*",
        type=int,
        help="number of iteration for EOT. (Default: 1)",
    )
    parser.add_argument(
        "--apgd_seed",
        default=0,
        nargs="*",
        type=int,
        help="random seed for the starting point. (Default: 0)",
    )
    parser.add_argument(
        "--apgd_steps",
        default=10,
        nargs="*",
        type=int,
        help="number of steps. (Default: 10)",
    )
    parser.add_argument(
        "--attack_iterations",
        type=int,
        default=iterations,
        nargs="*",
        help="Set number of iterations for adversarial attack.",
    )
    parser.add_argument(
        "--attack_alpha",
        type=float,
        default=alpha,
        help="Set alpha to use for adversarial attack.",
    )
    parser.add_argument(
        "--attack_targeted",
        type=attack_targeted_string,
        default=targeted,
        nargs="*",
        help="Set if adversarial attack should be targeted.",
    )
    parser.add_argument(
        "--attack_target",
        type=str,
        default="zero",
        nargs="*",
        choices=["zero", "negative"],
        help="Set the target for a tagreted attack.",
    )
    parser.add_argument(
        "--attack_loss",
        type=str,
        default="epe",
        nargs="*",
        help="Set the name of the used loss function (mse, epe)",
    )
    parser.add_argument(
        "--3dcc_intensity",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=3,
        nargs="*",
        help="Set the the intensity of the 3DCC corruption, int between 1 and 5",
    )
    parser.add_argument(
        "--3dcc_corruption",
        type=str,
        default="far_focus",
        nargs="*",
        choices=[
            "far_focus",
            "near_focus",
            "fog_3d",
            "color_quant",
            "iso_noise",
            "low_light",
            "xy_motion_blur",
            "z_motion_blur",
        ],
        help="Set the type of 3DCC",
    )
    parser.add_argument(
        "--selection",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Used in combination with model=select. The select mode can be used to run the validation on multiple models "
            "at once. Put a list of model names here separated by spaces."
        ),
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Used in combination with model=all. A list of model names that will not be validated."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(Path("outputs/validate")),
        help="Path to the directory where the validation results will be saved.",
    )
    parser.add_argument(
        "--write_outputs",
        action="store_true",
        help="If set, the estimated flow is saved to disk.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="If set, the results are shown on the screen.",
    )
    parser.add_argument(
        "--flow_format",
        type=str,
        default="original",
        choices=["flo", "png", "original"],
        help=(
            "The format to use when saving the estimated optical flow. If 'original', then the format will be the same "
            + "one the dataset uses for the groundtruth."
        ),
    )
    parser.add_argument(
        "--max_forward_side",
        type=int,
        default=None,
        help=(
            "If max(height, width) of the input image is larger than this value, then the image is downscaled "
            "before the forward and the outputs are bilinearly upscaled to the original resolution."
        ),
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=None,
        help=("Multiply the input image by this scale factor before forwarding."),
    )
    parser.add_argument(
        "--max_show_side",
        type=int,
        default=1000,
        help=(
            "If max(height, width) of the output image is larger than this value, then the image is downscaled "
            "before showing it on the screen."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help=(
            "Maximum number of samples per dataset will be used for calculating the metrics."
        ),
    )
    parser.add_argument(
        "--reversed",
        action="store_true",
        help="To be combined with model all or select. Iterates over the list of models in reversed order",
    )
    parser.add_argument(
        "--warm_start",
        action="store_true",
        help="If set, stores the previous estimation to be used a starting point for prediction.",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="If set, use half floating point precision."
    )
    parser.add_argument(
        "--seq_val_mode",
        type=str,
        default="all",
        choices=("all", "first", "middle", "last"),
        help=(
            "Used only when the model predicts outputs for more than one frame. Select which predictions will be used for evaluation."
        ),
    )
    parser.add_argument(
        "--write_individual_metrics",
        action="store_true",
        help="If set, save a table of metrics for every image.",
    )
    parser.add_argument("--overwrite_output", type=bool, default=False)
    return parser


def attack(args: Namespace, model: BaseModel) -> pd.DataFrame:
    """Perform the validation.

    Parameters
    ----------
    args : Namespace
        Arguments to configure the model and the validation.
    model : BaseModel
        The model to be used for validation.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the metric results.

    See Also
    --------
    ptlflow.models.base_model.base_model.BaseModel : The parent class of the available models.
    """
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        if args.fp16:
            model = model.half()
    dataloaders = model.val_dataloader()
    dataloaders = {
        model.val_dataloader_names[i]: dataloaders[i] for i in range(len(dataloaders))
    }
    overwrite_flag = args.overwrite_output
    output_data = []
    iteration_data = []
    start_time = datetime.now()
    output_data.append(("start_time", start_time.strftime("%Y-%m-%d %H:%M:%S")))
    output_data.append(("model", args.model))
    output_data.append(("checkpoint", args.pretrained_ckpt))
    iteration_data.append(("start_time", start_time.strftime("%Y-%m-%d %H:%M:%S")))
    iteration_data.append(("model", args.model))
    iteration_data.append(("checkpoint", args.pretrained_ckpt))
    attack_args_parser = AttackArgumentParser(args)

    for attack_args in attack_args_parser:
        for key, value in attack_args.items():
            if "_" in key:
                key = key.split("_")[1]
            if isinstance(value, float):
                value = round(value, 4)
            output_data.append((key, value))
            iteration_data.append((key, value))
        print(attack_args)
        for dataset_name, dl in dataloaders.items():
            metrics_mean, iteration_metrics_mean = attack_one_dataloader(
                args, attack_args, model, dl, dataset_name
            )

            end_time = datetime.now()
            output_data.append(("end_time", end_time.strftime("%Y-%m-%d %H:%M:%S")))
            iteration_data.append(("end_time", end_time.strftime("%Y-%m-%d %H:%M:%S")))
            time_difference = end_time - start_time
            hours = time_difference.seconds // 3600
            minutes = (time_difference.seconds % 3600) // 60
            seconds = time_difference.seconds % 60
            time_difference_str = "{:02}:{:02}:{:02}".format(hours, minutes, seconds)
            output_data.append(("duration", time_difference_str))
            iteration_data.append(("duration", time_difference_str))

            output_data.append(("dataset", dataset_name))
            iteration_data.append(("dataset", dataset_name))
            for k in metrics_mean.keys():
                output_data.append((k, metrics_mean[k]))
            for k in iteration_metrics_mean.keys():
                iteration_data.append((k, iteration_metrics_mean[k]))

            args.output_path.mkdir(parents=True, exist_ok=True)

            output_dict = {}
            for key, value in output_data:
                if "val" in key:
                    output_dict.setdefault("metrics", {})[key.split("/")[1]] = value
                else:
                    output_dict[key] = value
            output_filename = args.output_path / f"metrics_{args.val_dataset}.json"

            if os.path.exists(output_filename) and not overwrite_flag:
                with open(output_filename, "r") as json_file:
                    metrics = json.load(json_file)
            else:
                metrics = {"experiments": []}

            metrics["experiments"].append(output_dict)

            with open(output_filename, "w") as json_file:
                json.dump(metrics, json_file, indent=4)

            if iteration_metrics_mean:
                iteration_output_dict = {}
                for key, value in iteration_data:
                    if "val" in key:
                        iteration_output_dict.setdefault("metrics", {})[
                            key.split("/")[1]
                        ] = value
                    else:
                        iteration_output_dict[key] = value
                output_filename = (
                    args.output_path / f"iteration_metrics_{args.val_dataset}.json"
                )

                if os.path.exists(output_filename) and not overwrite_flag:
                    with open(output_filename, "r") as json_file:
                        metrics = json.load(json_file)
                else:
                    metrics = {"experiments": []}

                metrics["experiments"].append(iteration_output_dict)

                with open(output_filename, "w") as json_file:
                    json.dump(metrics, json_file, indent=4)


def attack_list_of_models(args: Namespace) -> None:
    """Perform the validation.

    Parameters
    ----------
    args : Namespace
        Arguments to configure the list of models and the validation.
    """
    metrics_df = pd.DataFrame()

    model_names = _get_model_names(args)
    if args.reversed:
        model_names = reversed(model_names)

    exclude = args.exclude
    if exclude is None:
        exclude = []
    for mname in model_names:
        if mname in exclude:
            continue

        logging.info(mname)
        model_ref = get_model_reference(mname)

        if hasattr(model_ref, "pretrained_checkpoints"):
            ckpt_names = model_ref.pretrained_checkpoints.keys()
            for cname in ckpt_names:
                try:
                    logging.info(cname)
                    parser_tmp = model_ref.add_model_specific_args(parser)
                    args = parser_tmp.parse_args()

                    args.model = mname
                    args.pretrained_ckpt = cname

                    model_id = args.model
                    if args.pretrained_ckpt is not None:
                        model_id += f"_{args.pretrained_ckpt}"
                    args.output_path = Path(args.output_path) / model_id

                    model = get_model(mname, cname, args)
                    instance_metrics_df = attack(args, model)
                    metrics_df = pd.concat([metrics_df, instance_metrics_df])
                    args.output_path.parent.mkdir(parents=True, exist_ok=True)
                    if args.reversed:
                        metrics_df.to_csv(
                            args.output_path.parent / "metrics_all_rev.csv", index=False
                        )
                    else:
                        metrics_df.to_csv(
                            args.output_path.parent / "metrics_all.csv", index=False
                        )
                except Exception as e:  # noqa: B902
                    logging.warning("Skipping model %s due to exception %s", mname, e)
                    break


@torch.enable_grad()
def attack_one_dataloader(
    args: Namespace,
    attack_args: Dict[str, List[object]],
    model: BaseModel,
    dataloader: DataLoader,
    dataloader_name: str,
) -> Dict[str, float]:
    """Perform adversarial attack for all examples of one dataloader.

    Parameters
    ----------
    args : Namespace
        Arguments to configure the model and the validation.
    model : BaseModel
        The model to be used for validation.
    dataloader : DataLoader
        The dataloader for the validation.
    dataloader_name : str
        A string to identify this dataloader.

    Returns
    -------
    Dict[str, float]
        The average metric values for this dataloader.
    """
    metrics_sum = {}
    iteration_metrics_sum = {}
    if attack_args["attack"] == "3dcc":
        dataloader = get_dataset_3DCC(
            model,
            dataloader_name,
            attack_args["3dcc_corruption"],
            attack_args["3dcc_intensity"],
        )
    metrics_individual = None
    if args.write_individual_metrics:
        metrics_individual = {"filename": [], "epe": [], "outlier": []}

    with tqdm(dataloader) as tdl:
        prev_preds = None
        for i, inputs in enumerate(tdl):
            if args.scale_factor is not None:
                scale_factor = args.scale_factor
            else:
                scale_factor = (
                    None
                    if args.max_forward_side is None
                    else float(args.max_forward_side) / min(inputs["images"].shape[-2:])
                )

            io_adapter = IOAdapter(
                model,
                inputs["images"].shape[-2:],
                target_scale_factor=scale_factor,
                cuda=torch.cuda.is_available(),
                fp16=args.fp16,
            )
            inputs = io_adapter.prepare_inputs(inputs=inputs, image_only=True)
            inputs["prev_preds"] = prev_preds

            if inputs["images"].max() > 1.0:
                attack_args["attack_epsilon"] = attack_args["attack_epsilon"] * 255
                attack_args["attack_alpha"] = attack_args["attack_alpha"] * 255
            has_ground_truth = True
            targeted_inputs = None

            with torch.no_grad():
                orig_preds = model(inputs)
            torch.cuda.empty_cache()

            if attack_args["attack_targeted"] or attack_args["attack"] == "pcfa":
                if attack_args["attack_target"] == "negative":
                    targeted_flow_tensor = -orig_preds["flows"]
                else:
                    targeted_flow_tensor = torch.zeros_like(orig_preds["flows"])
                if not "flows" in inputs:
                    inputs["flows"] = targeted_flow_tensor
                    has_ground_truth = False

                targeted_inputs = inputs.copy()
                targeted_inputs["flows"] = targeted_flow_tensor
            # for logging
            if attack_args["attack"] == "none":
                targeted_inputs = inputs.copy()

            iteration_metrics = {}
            # pdb.set_trace()
            match attack_args["attack"]:  # Commit adversarial attack
                case "fgsm":
                    (
                        preds,
                        perturbed_inputs,
                    ) = fgsm(attack_args, inputs, model, targeted_inputs)
                case "pgd":
                    preds, perturbed_inputs, iteration_metrics = bim_pgd_cospgd(
                        attack_args, inputs, model, targeted_inputs, orig_preds
                    )
                case "cospgd":
                    preds, perturbed_inputs, iteration_metrics = bim_pgd_cospgd(
                        attack_args, inputs, model, targeted_inputs, orig_preds
                    )
                case "bim":
                    preds, perturbed_inputs, iteration_metrics = bim_pgd_cospgd(
                        attack_args, inputs, model, targeted_inputs, orig_preds
                    )
                case "apgd":
                    preds, perturbed_inputs = apgd(
                        attack_args, inputs, model, targeted_inputs
                    )
                case "fab":
                    preds, perturbed_inputs = fab(
                        attack_args, inputs, model, targeted_inputs
                    )
                case "pcfa":
                    preds, perturbed_inputs, iteration_metrics = pcfa(
                        attack_args, inputs, model, targeted_inputs
                    )
                case "common_corruptions":
                    preds, perturbed_inputs = common_corrupt(attack_args, inputs, model)
                case "none":
                    preds = orig_preds

            for key in preds:
                if torch.is_tensor(preds[key]):
                    preds[key] = preds[key].detach()
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].detach()
            if attack_args["attack"] != "none":
                for key in perturbed_inputs:
                    if torch.is_tensor(perturbed_inputs[key]):
                        perturbed_inputs[key] = perturbed_inputs[key].detach()
            for key in iteration_metrics:
                if torch.is_tensor(iteration_metrics[key]):
                    iteration_metrics[key] = iteration_metrics[key].detach()

            if args.warm_start:
                if (
                    "is_seq_start" in inputs["meta"]
                    and inputs["meta"]["is_seq_start"][0]
                ):
                    prev_preds = None
                else:
                    prev_preds = preds
                    for k, v in prev_preds.items():
                        if isinstance(v, torch.Tensor):
                            prev_preds[k] = v.detach()

            inputs = io_adapter.unscale(inputs, image_only=True)
            preds = io_adapter.unscale(preds)

            if attack_args["attack"] != "none":
                perturbed_inputs = io_adapter.unscale(perturbed_inputs, image_only=True)
                if attack_args["attack_targeted"] or attack_args["attack"] == "pcfa":
                    targeted_inputs = io_adapter.unscale(
                        targeted_inputs, image_only=True
                    )

            if inputs["flows"].shape[1] > 1 and args.seq_val_mode != "all":
                if args.seq_val_mode == "first":
                    k = 0
                elif args.seq_val_mode == "middle":
                    k = inputs["images"].shape[1] // 2
                elif args.seq_val_mode == "last":
                    k = inputs["flows"].shape[1] - 1
                for key, val in inputs.items():
                    if key == "meta":
                        inputs["meta"]["image_paths"] = inputs["meta"]["image_paths"][
                            k : k + 1
                        ]
                    elif key == "images":
                        inputs[key] = val[:, k : k + 2]
                    elif isinstance(val, torch.Tensor) and len(val.shape) == 5:
                        inputs[key] = val[:, k : k + 1]

            if attack_args["attack_targeted"] or attack_args["attack"] == "pcfa":

                metrics = model.val_metrics(preds, targeted_inputs)

                criterion = LossCriterion("epe")
                loss = criterion.loss(
                    preds["flows"].squeeze(0).float(),
                    targeted_inputs["flows"].squeeze(0).float(),
                )
                loss = loss.mean()
                metrics["val/own_epe"] = loss

                metrics_orig_preds = model.val_metrics(preds, orig_preds)
                metrics["val/epe_orig_preds"] = metrics_orig_preds["val/epe"]

                loss = criterion.loss(
                    preds["flows"].squeeze(0).float(),
                    orig_preds["flows"].squeeze(0).float(),
                )
                loss = loss.mean()
                metrics["val/own_epe_orig_preds"] = loss

                metrics["val/cosim_target"] = torch.mean(
                    cosine_similarity(
                        get_flow_tensors(preds), get_flow_tensors(targeted_inputs)
                    )
                )
                metrics["val/cosim_orig_preds"] = torch.mean(
                    cosine_similarity(
                        get_flow_tensors(preds), get_flow_tensors(orig_preds)
                    )
                )
                if has_ground_truth:
                    metrics_ground_truth = model.val_metrics(preds, inputs)
                    metrics["val/epe_ground_truth"] = metrics_ground_truth["val/epe"]

                    loss = criterion.loss(
                        preds["flows"].squeeze(0).float(),
                        inputs["flows"].squeeze(0).float(),
                    )
                    loss = loss.mean()
                    metrics["val/own_epe_ground_truth"] = loss

                    metrics["val/cosim_ground_truth"] = torch.mean(
                        cosine_similarity(
                            get_flow_tensors(preds), get_flow_tensors(inputs)
                        )
                    )
            else:
                metrics = model.val_metrics(preds, inputs)

                criterion = LossCriterion("epe")
                loss = criterion.loss(
                    preds["flows"].squeeze(0).float(),
                    inputs["flows"].squeeze(0).float(),
                )
                loss = loss.mean()
                metrics["val/own_epe"] = loss

                metrics["val/cosim"] = torch.mean(
                    cosine_similarity(get_flow_tensors(preds), get_flow_tensors(inputs))
                )

                if attack_args["attack"] == "none":
                    targeted_flow_tensor_negative = -orig_preds["flows"].clone()
                    targeted_flow_tensor_zero = torch.zeros_like(orig_preds["flows"])
                    loss_initial_neg = criterion.loss(
                        preds["flows"].squeeze(0).float(),
                        targeted_flow_tensor_negative.squeeze(0).float(),
                    )
                    loss_initial_neg = loss_initial_neg.mean()
                    loss_initial_zero = criterion.loss(
                        preds["flows"].squeeze(0).float(),
                        targeted_flow_tensor_zero.squeeze(0).float(),
                    )
                    loss_initial_zero = loss_initial_zero.mean()
                    metrics["val/epe_initial_to_negative"] = loss_initial_neg
                    metrics["val/epe_initial_to_zero"] = loss_initial_zero

                    loss_ground_truth_neg = criterion.loss(
                        inputs["flows"].squeeze(0).float(),
                        targeted_flow_tensor_negative.squeeze(0).float(),
                    )
                    loss_ground_truth_neg = loss_ground_truth_neg.mean()
                    loss_ground_truth_zero = criterion.loss(
                        inputs["flows"].squeeze(0).float(),
                        targeted_flow_tensor_zero.squeeze(0).float(),
                    )
                    loss_ground_truth_zero = loss_ground_truth_zero.mean()
                    metrics["val/own_epe_ground_truth_to_negative"] = (
                        loss_ground_truth_neg
                    )
                    metrics["val/own_epe_ground_truth_to_zero"] = loss_ground_truth_zero

                    targeted_inputs["flows"] = targeted_flow_tensor_negative.float()

                    metrics_ground_truth_negative = model.val_metrics(
                        targeted_inputs, inputs
                    )
                    metrics["val/epe_ground_truth_to_negative"] = (
                        metrics_ground_truth_negative["val/epe"]
                    )

                    targeted_inputs["flows"] = targeted_flow_tensor_zero.float()
                    metrics_ground_truth_zero = model.val_metrics(
                        targeted_inputs, inputs
                    )
                    metrics["val/epe_ground_truth_to_zero"] = metrics_ground_truth_zero[
                        "val/epe"
                    ]
            if attack_args["attack"] != "none":
                adv_image1, adv_image2 = get_image_tensors(perturbed_inputs)
                image1, image2 = get_image_tensors(inputs)
                delta1 = adv_image1 - image1
                delta2 = adv_image2 - image2
                delta_dic = losses.calc_delta_metrics(delta1, delta2)
                for k, v in delta_dic.items():
                    metrics[k] = v

            for k in metrics.keys():
                if metrics_sum.get(k) is None:
                    metrics_sum[k] = 0.0
                metrics_sum[k] += metrics[k].item()

            for k in iteration_metrics.keys():
                if iteration_metrics_sum.get(k) is None:
                    iteration_metrics_sum[k] = 0.0
                iteration_metrics_sum[k] += iteration_metrics[k].item()

            
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                tdl.set_postfix(
                    epe=metrics_sum["val/epe"] / (i + 1),
                    outlier=metrics_sum["val/outlier"] / (i + 1),
                    total=total,
                    free=free,
                )
            else:
                tdl.set_postfix(
                    epe=metrics_sum["val/epe"] / (i + 1),
                    outlier=metrics_sum["val/outlier"] / (i + 1),
                )

            filename = ""
            if "sintel" in inputs["meta"]["dataset_name"][0].lower():
                filename = f'{Path(inputs["meta"]["image_paths"][0][0]).parent.name}/'
            filename += Path(inputs["meta"]["image_paths"][0][0]).stem

            if metrics_individual is not None:
                metrics_individual["filename"].append(filename)
                metrics_individual["epe"].append(metrics["val/epe"].item())
                metrics_individual["outlier"].append(metrics["val/outlier"].item())

            if attack_args["attack"] != "none":
                generate_outputs(
                    args,
                    preds,
                    dataloader_name,
                    i,
                    inputs.get("meta"),
                    perturbed_inputs,
                    attack_args,
                )
            else:
                generate_outputs(args, preds, dataloader_name, i, inputs.get("meta"))
            if args.max_samples is not None and i >= (args.max_samples - 1):
                break

            del preds
            del inputs
            del iteration_metrics
            if attack_args["attack"] != "none":
                del perturbed_inputs
            torch.cuda.empty_cache()

    if args.write_individual_metrics:
        ind_df = pd.DataFrame(metrics_individual)
        args.output_path.mkdir(parents=True, exist_ok=True)
        ind_df.to_csv(
            Path(args.output_path) / f"{dataloader_name}_epe_outlier.csv", index=None
        )

    metrics_mean = {}
    for k, v in metrics_sum.items():
        metrics_mean[k] = v / len(dataloader)

    iteration_metrics_mean = {}
    for k, v in iteration_metrics_sum.items():
        iteration_metrics_mean[k] = v / len(dataloader)

    return metrics_mean, iteration_metrics_mean


def generate_outputs(
    args: Namespace,
    preds: Dict[str, torch.Tensor],
    dataloader_name: str,
    batch_idx: int,
    metadata: Optional[Dict[str, Any]] = None,
    perturbed_inputs: Optional[Dict[str, torch.Tensor]] = None,
    attack_args: Optional[Dict[str, List[object]]] = None,
) -> None:
    """Display on screen and/or save outputs to disk, if required.

    Parameters
    ----------
    args : Namespace
        The arguments with the required values to manage the outputs.
    inputs : Dict[str, torch.Tensor]
        The inputs loaded from the dataset (images, groundtruth).
    preds : Dict[str, torch.Tensor]
        The model predictions (optical flow and others).
    dataloader_name : str
        A string to identify from which dataloader these inputs came from.
    batch_idx : int
        Indicates in which position of the loader this input is.
    metadata : Dict[str, Any], optional
        Metadata about this input, if available.
    """

    preds = tensor_dict_to_numpy(preds)
    preds["flows_viz"] = flow_utils.flow_to_rgb(preds["flows"])[:, :, ::-1]
    if preds.get("flows_b") is not None:
        preds["flows_b_viz"] = flow_utils.flow_to_rgb(preds["flows_b"])[:, :, ::-1]

    if args.write_outputs:
        if perturbed_inputs is not None:
            # perturbed_inputs = tensor_dict_to_numpy(perturbed_inputs)
            _write_to_npy_file(
                args,
                preds,
                dataloader_name,
                batch_idx,
                metadata,
                perturbed_inputs,
                attack_args,
            )
        else:
            _write_to_npy_file(args, preds, dataloader_name, batch_idx, metadata)


def _write_to_npy_file(
    args: Namespace,
    preds: Dict[str, torch.Tensor],
    dataloader_name: str,
    batch_idx: int,
    metadata: Optional[Dict[str, Any]] = None,
    perturbed_inputs: Dict[str, torch.Tensor] = None,
    attack_args: Optional[Dict[str, List[object]]] = None,
) -> None:

    out_root_dir = Path(args.output_path) / dataloader_name
    extra_dirs = ""
    if metadata is not None:
        img_path = Path(metadata["image_paths"][0][0])
        img2_path = Path(metadata["image_paths"][1][0])
        image_name = img_path.stem
        image2_name = img2_path.stem
        if "sintel" in dataloader_name:
            seq_name = img_path.parts[-2]
            extra_dirs = seq_name
        elif "spring" in dataloader_name:
            seq_name = img_path.parts[-3]
            extra_dirs = seq_name
    else:
        image_name = f"{batch_idx:08d}"
        image2_name = f"{batch_idx:08d}_2"

    if args.flow_format != "original":
        flow_ext = args.flow_format
    else:
        if "kitti" in dataloader_name or "hd1k" in dataloader_name or "sintel" in dataloader_name or "spring" in dataloader_name:
            flow_ext = "png"
        else:
            flow_ext = "flo"

    for k, v in preds.items():
        if isinstance(v, np.ndarray):
            out_dir = out_root_dir
            if perturbed_inputs is not None:
                for arg, val in attack_args.items():
                    out_dir = out_dir / f"{arg}={val}"

        if flow_ext == "png":
            if k == "flows_viz":
                out_dir_flows = out_dir / k / extra_dirs
                out_dir_flows.mkdir(parents=True, exist_ok=True)
                # np.savez_compressed(
                #     str(out_dir_flows / f"{image_name}"), v.astype(np.uint8)
                # )
                cv.imwrite(
                    str(out_dir_flows / f"{image_name}.{flow_ext}"), v.astype(np.uint8)
                )
        elif k == "flows":
            out_dir_flows = out_dir / k / extra_dirs
            out_dir_flows.mkdir(parents=True, exist_ok=True)
            flow_utils.flow_write(out_dir_flows / f"{image_name}.{flow_ext}", v)

    if perturbed_inputs is not None:
        for k, v in perturbed_inputs.items():
            if k == "images":
                out_dir_imgs = out_dir / k / extra_dirs
                out_dir_imgs.mkdir(parents=True, exist_ok=True)
                if v.max() <= 1:
                    v = v * 255

                image = v[0, 0].detach().cpu()
                image2 = v[0, 1].detach().cpu()
                # Convert from (C, H, W) to (H, W, C)
                image = image.permute(1, 2, 0).numpy()
                image2 = image2.permute(1, 2, 0).numpy()
                output_filepath = out_dir_imgs / f"{image_name}.png"
                output_filepath2 = out_dir_imgs / f"{image2_name}.png"

                cv.imwrite(str(output_filepath), image.astype(np.uint8))
                cv.imwrite(str(output_filepath2), image2.astype(np.uint8))


def _write_to_file(
    args: Namespace,
    preds: Dict[str, torch.Tensor],
    dataloader_name: str,
    batch_idx: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    out_root_dir = Path(args.output_path) / dataloader_name
    # pdb.set_trace()
    extra_dirs = ""
    if metadata is not None:
        img_path = Path(metadata["image_paths"][0][0])
        image_name = img_path.stem
        if "sintel" in dataloader_name:
            seq_name = img_path.parts[-2]
            extra_dirs = seq_name
    else:
        image_name = f"{batch_idx:08d}"

    if args.flow_format != "original":
        flow_ext = args.flow_format
    else:
        if "kitti" in dataloader_name or "hd1k" in dataloader_name:
            flow_ext = "png"
        else:
            flow_ext = "flo"

    for k, v in preds.items():
        if isinstance(v, np.ndarray):
            out_dir = out_root_dir / k / extra_dirs
            out_dir.mkdir(parents=True, exist_ok=True)
            if k == "flows" or k == "flows_b":
                flow_utils.flow_write(out_dir / f"{image_name}.{flow_ext}", v)
            elif len(v.shape) == 2 or (
                len(v.shape) == 3 and (v.shape[2] == 1 or v.shape[2] == 3)
            ):
                if v.max() <= 1:
                    v = v * 255
                # pdb.set_trace()
                cv.imwrite(str(out_dir / f"{image_name}.png"), v.astype(np.uint8))


if __name__ == "__main__":
    parser = _init_parser()

    # TODO: It is ugly that the model has to be gotten from the argv rather than the argparser.
    # However, I do not see another way, since the argparser requires the model to load some of the args.
    FlowModel = None
    if len(sys.argv) > 1 and sys.argv[1] not in ["-h", "--help", "all", "select"]:
        FlowModel = get_model_reference(sys.argv[1])
        parser = FlowModel.add_model_specific_args(parser)

    add_datasets_to_parser(parser, "./ptlflow/datasets.yml")

    args = parser.parse_args()

    if args.model not in ["all", "select"]:
        model_id = args.model
        if args.pretrained_ckpt is not None:
            model_id += f"_{Path(args.pretrained_ckpt).stem}"
        if args.max_forward_side is not None:
            model_id += f"_maxside{args.max_forward_side}"
        if args.scale_factor is not None:
            model_id += f"_scale{args.scale_factor}"
        args.output_path = Path(args.output_path) / model_id
        model = get_model(sys.argv[1], args.pretrained_ckpt, args)
        args.output_path.mkdir(parents=True, exist_ok=True)

        attack(args, model)
    else:
        attack_list_of_models(args)


def evaluate(model_name, dataset, pretrained_ckpt, dataset_path=None, threat_model="none", iterations=20, epsilon=8/255, alpha=0.01, lp_norm="inf", retrieve_existing=False, output_path="./output", write_outputs=False, dataset_config_path=None, additional_args=None):
    """
    Function to run an attack based on the specified parameters.
    
    Args:
        model_name (str): The name of the model to attack.
        dataset (str): The dataset on which the model will be attacked.
        pretrained_ckpt (str): Path to the pretrained model checkpoint.
        output_path (str): Path to save the attack results.
        attack_name (str): Specific attack to use (if applicable).
        write_outputs (bool): Whether to save attack outputs.
        dataset_config (str): Path to the dataset config file (default: "./ptlflow/datasets.yml").
        additional_args (dict): Any additional arguments that would be needed by the attack or model.
    """

    # Step 1: Fetch model reference based on the model name
    FlowModel = get_model_reference(model_name)

    # Step 2: Initialize the parser and add model-specific args
    parser = _init_parser()
    parser = FlowModel.add_model_specific_args(parser)
    
    # Create a list of parameters to simulate CLI arguments
    list_of_parameters = [
        f"--model={model_name}",
        f"--val_dataset={dataset}",
        f"--output_path={Path(output_path)} / {model_name}_{Path(pretrained_ckpt).stem}",
        f"--attack={threat_model}",
        f"--pretrained_ckpt={pretrained_ckpt}",
    ]
    if write_outputs:
        list_of_parameters.append("--write_outputs")

    # Add additional args from dictionary
    # TODO: remove attack_ from all the args of the parser and from the attack_args_parser.py
    if additional_args:
        for key, value in additional_args.items():
            if key in ["epsilon", "alpha", "lp_norm", "iterations"]:
                if key == "lp_norm":
                    list_of_parameters.append(f"--attack_norm={value}")
                else:
                    list_of_parameters.append(f"--attack_{key}={value}")
            else:
                list_of_parameters.append(f"--{key}={value}")

    # Step 3: Parse the arguments using the parameter list
    args = parser.parse_args(list_of_parameters)

    # Step 4: Add dataset-specific arguments manually
    if dataset_config_path:
        with open(dataset_config_path, "r") as f:
            dataset_paths = yaml.safe_load(f)
        for name, path in dataset_paths.items():
            setattr(args, f"{name}_root_dir", path)
    elif dataset_path:
        if dataset == "kitti-2015":
            setattr(args, "kitti_2015_root_dir", dataset_path)
        elif dataset == "kitti-2012":
            setattr(args, "kitti_2012_root_dir", dataset_path)
        elif "sintel" in dataset:
            setattr(args, "mpi_sintel_root_dir", dataset_path)
        elif "spring" in dataset:
            setattr(args, "spring_root_dir", dataset_path)
        elif dataset == "things":
            setattr(args, "flying_things3d", dataset_path)
    else:
        if dataset == "kitti-2015":
            setattr(args, "kitti_2015_root_dir", "./datasets/kitti2015")
        elif dataset == "kitti-2012":
            setattr(args, "kitti_2012_root_dir", "./datasets/kitti2012")
        elif "sintel" in dataset:
            setattr(args, "mpi_sintel_root_dir", "./datasets/Sintel")
        elif "spring" in dataset:
            setattr(args, "spring_root_dir", "./datasets/spring")
        elif dataset == "things":
            setattr(args, "flying_things3d", "./datasets/flyingthings3d")


    # TODO: Try to retrieve existing results
    df = retrieve_data(args)
    if df:
        return df


    # Step 5: Ensure output directory exists
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    model = get_model(model_name, pretrained_ckpt, args)

    # Run the attack on the specific model
    attack(args, model)


# TODO:
def retrieve_data(args):
    attack_args_parser = AttackArgumentParser(args)
    modified_attack_args = {}
    for attack_args in attack_args_parser:
        for key, value in attack_args.items():
            if "_" in key:
                key = key.split("_")[1]
            if isinstance(value, float):
                value = round(value, 4)
            modified_attack_args[key] = value
    print("Retrieving data for:")
    print(modified_attack_args)

    return None

def load_model(model_name, dataset):
    model_ref = get_model_reference(model_name)
    checkpoints = model_ref.pretrained_checkpoints.keys()
    for c in checkpoints:
        if c in dataset:
            model = get_model(model_name, c)
            return model
    print(f"No pre-trained model available for {model_name}/{dataset}.")
    return None