# coding=utf-8
import os
import time
from typing import Dict, List

from ptlflow.models.base_model.base_model import BaseModel
from attacks.attack_utils.utils import (
    get_image_tensors,
    get_flow_tensors,
    replace_images_dic
)

from attacks.help_function import weather_model
from attacks.help_function.render import render
from attacks.help_function.weather import get_weather
from attacks.help_function.logging import save_image, save_flow
from attacks.help_function.sintel_io import cam_read, depth_read
from attacks.help_function.datasets import rescale_sintel_scenes

import torch
import attacks.attack_utils.loss_criterion as losses
import torch.optim as optim
import numpy as np


def get_optimizer(optimizer_name, optimization_parameters, optimizer_lr=0.001):
    if optimizer_name == "Adam":
        optimizer = optim.Adam(optimization_parameters, lr=optimizer_lr)
    else:
        raise ValueError(
            "The selected optimizer option '%s' is unknown. Select 'Adam'."
            % optimizer_name
        )
    return optimizer


def weather(
    attack_args: Dict[str, List[object]],
    model: BaseModel,
    targeted_inputs: Dict[str, torch.Tensor],
    batch_index: int,
    output_path: str,
):
    torch.cuda.empty_cache()
    print("Print all keys", attack_args)

    # substitude model specific args
    if attack_args["model"] in ("gma", "raft"):
        model.args.iters = attack_args["weather_model_iters"]
        

    # Define what device we are using
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    print("Device now is：", device)

    # Get Related cam/depth data and weather data
    meta_info = targeted_inputs["meta"]
    scene = meta_info["misc"][0]
    image_path = meta_info["image_paths"][0][0]
    image_index = int(image_path.split(".")[-2][-4:])
    ds_prefix = image_path.split(os.path.sep)[:-2]

    cam_left_root = os.path.sep.join(ds_prefix[:-1] + ["camdata_left"])
    depth_root = os.path.sep.join(ds_prefix[:-1] + ["depth"])

    weather_datum_path = os.path.join(
        attack_args["weather_data"], scene, f"frame_{image_index:04d}.npz"
    )

    # cam data
    a_int, a_ext_1 = cam_read(
        os.path.join(cam_left_root, f"{scene}/frame_{image_index:04d}.cam")
    )
    _, a_ext_2 = cam_read(
        os.path.join(cam_left_root, f"{scene}/frame_{image_index+1:04d}.cam")
    )
    a_ext_1 = np.vstack((a_ext_1, np.array([[0, 0, 0, 1]])))
    a_ext_2 = np.vstack((a_ext_2, np.array([[0, 0, 0, 1]])))

    full_p = np.vstack((np.hstack((a_int, [[0], [0], [0]])), [[0, 0, 0, 1]]))
    full_p = torch.from_numpy(full_p).float()
    rel_mat = torch.from_numpy(a_ext_2 @ np.linalg.inv(a_ext_1)).float()

    # depth data
    gt1_depth = depth_read(
        os.path.join(depth_root, f"{scene}/frame_{image_index:04d}.dpt")
    )
    gt1_depth = torch.unsqueeze(torch.from_numpy(gt1_depth), dim=0)
    gt2_depth = depth_read(
        os.path.join(depth_root, f"{scene}/frame_{image_index+1:04d}.dpt")
    )
    gt2_depth = torch.unsqueeze(torch.from_numpy(gt2_depth), dim=0)

    gt1_depth, gt2_depth, rel_mat = rescale_sintel_scenes(
        scene,
        gt1_depth,
        gt2_depth,
        rel_mat,
        OVERALL_SCALE=attack_args["weather_scene_scale"],
    )

    ext_1 = torch.from_numpy(a_ext_1)

    scene_data = tuple(
        [
            torch.stack((item,), 0)
            for item in (full_p, ext_1, rel_mat, gt1_depth, gt2_depth)
        ]
    )

    # weather data
    weather = get_weather(
        has_weather=True,
        weatherdat=weather_datum_path,
        scene_data=scene_data,
        args=attack_args,
        seed=None,
        load_only=True,
    )

    scene_data = [i.to(device) for i in scene_data]
    weather = [i.to(device) for i in weather]

    preds, image1_weather, image2_weather = attack_image(
        model,
        targeted_inputs,
        attack_args,
        device,
        scene_data,
        weather,
        batch_index,
        output_path,
        image_index,
        scene,
    )
    print("preds", preds)

    perturbed_inputs = replace_images_dic(targeted_inputs, image1_weather, image2_weather)

    return preds, perturbed_inputs


def attack_image(
    model,
    targeted_inputs,
    attack_args,
    device,
    scene_data,
    weather,
    batch_index,
    output_path,
    image_index,
    scene,
):

    # torch.autograd.set_detect_anomaly(True)
    model = weather_model.ScaledInputWeatherModel(model)

    image1, image2 = get_image_tensors(targeted_inputs)
    image1, image2 = image1.to(device), image2.to(device)

    # bgr2rgb
    image1 = image1[:, [2, 1, 0], :, :]
    image2 = image2[:, [2, 1, 0], :, :]

    # pdb.set_trace()
    flow = get_flow_tensors(targeted_inputs)
    flow = flow.to(device)

    # Set requires_grad attribute of tensor. Important for Attack
    image1.requires_grad = False
    image2.requires_grad = False

    eps_box = 1e-7
    full_P, ext1, rel_mat, gt1_depth, gt2_depth = scene_data

    # Set requires_grad attribute of tensor. Important for Attack
    image1.requires_grad = False
    image2.requires_grad = False
    gt1_depth.requires_grad = False
    gt2_depth.requires_grad = False
    full_P.requires_grad = False
    ext1.requires_grad = False
    rel_mat.requires_grad = False

    scene_data = full_P, ext1, rel_mat, gt1_depth, gt2_depth
    initpos, motion, flakes, flakes_color, flakes_transp = weather

    # ===== ATTACK OPTIMIZATION =====

    torch.autograd.set_detect_anomaly(True)

    # prepare and rescale images
    offsets = torch.zeros_like(initpos).detach().clone()
    motion_offsets = torch.zeros_like(initpos)

    flow_weather_pred_init = None
    flakes_transp_init = flakes_transp.clone().detach()

    # non-learnable parameters
    initpos.requires_grad = False
    flakes.requires_grad = False
    motion.requires_grad = False
    flakes_color.requires_grad = False
    flakes_color_inf = torch.atanh(
        2.0 * (1.0 - eps_box) * (flakes_color) - (1 - eps_box)
    )  # switch color to [-infty, infty] for better optimization
    # Note: the transparencies are already atanh-transformed

    # learnable parameters
    offsets.requires_grad = True
    motion_offsets.requires_grad = True
    flakes_transp.requires_grad = True
    flakes_color_inf.requires_grad = True

    learnparams = []
    if attack_args["weather_learn_offset"]:
        learnparams += [offsets]
    if attack_args["weather_learn_motionoffset"]:
        learnparams += [motion_offsets]
    if attack_args["weather_learn_transparency"]:
        learnparams += [flakes_transp]
    if attack_args["weather_learn_color"]:
        learnparams += [flakes_color_inf]
    if learnparams == []:
        raise ValueError(
            "No learnable parameters were passed in the argument parser. Cannot optimize particles."
        )

    optimizer = get_optimizer(
        attack_args["weather_optimizer"], learnparams, optimizer_lr=0.001
    )

    # Predict the flow
    flakes_color_img = (
        (1.0 / 2.0)
        * 1.0
        / (1.0 - eps_box)
        * (torch.tanh(flakes_color_inf) + (1 - eps_box))
    )
    weather = (
        initpos + offsets.clone().detach(),
        motion + motion_offsets.clone().detach(),
        flakes,
        flakes_color_img.clone().detach(),
        flakes_transp.clone().detach(),
    )

    # rendered_image1, rendered_image2 = render(image1, image2, scene_data, weather, args=attack_args)
    # perturbed_inputs = replace_images_dic(targeted_inputs, rendered_image1, rendered_image2, clone=True)


    # # define the initial flow, the target, and update mu
    # flow_weather_pred_init = flow_weather_pred.detach().clone()
    # flow_weather_pred_init.requires_grad = False

    # with torch.no_grad():
    #     preds_init = model(
    #         image1, image2, weather=None, scene_data=None, args_=attack_args
    #     )
    # flow_init = preds_init["flows"].squeeze(0)
    # flow_init = flow_init.to(device).detach()


    with torch.no_grad():
        preds = model(
            image1, image2, weather=weather, scene_data=scene_data, args_=attack_args
        )
    torch.cuda.empty_cache()
    flow_weather_pred = preds["flows"].squeeze(0)
    flow_weather_pred = flow_weather_pred.to(device)

    # 调换顺序至257-266 define the initial flow, the target, and update mu
    flow_weather_pred_init = flow_weather_pred.detach().clone()
    flow_weather_pred_init.requires_grad = False

    with torch.no_grad():
        preds_init = model(
            image1, image2, weather=None, scene_data=None, args_=attack_args
        )
    flow_init = preds_init["flows"].squeeze(0)
    flow_init = flow_init.to(device).detach()

    target = get_flow_tensors(targeted_inputs)
    target = target.to(device)
    target.requires_grad = False

    # flow_weather_min = flow_weather_pred_init.clone().detach()
    offsets_min = offsets.clone().detach()
    motion_offsets_min = motion_offsets.clone().detach()
    flakes_color_min = flakes_color_img.clone().detach()
    flakes_transp_min = flakes_transp.clone().detach()

    # Zero all existing gradients
    model.zero_grad()
    optimizer.zero_grad()

    for steps in range(attack_args["weather_steps"]):
        print(f"into Schleife, step {steps}")
        # Calculate loss
        loss = losses.loss_weather(
            flow_weather_pred,
            target,
            f_type=attack_args["attack_loss"],
            init_pos=initpos,
            offsets=offsets,
            motion_offsets=motion_offsets,
            flakes_transp=flakes_transp,
            flakes_transp_init=flakes_transp_init,
            alph_offsets=attack_args["weather_alph_motion"],
            alph_motion=attack_args["weather_alph_motionoffset"],
            alph_transp=0,
        )

        loss.backward(retain_graph=True)

        if attack_args["weather_optimizer"] in ["Adam"]:
            optimizer.step()
        else:
            raise RuntimeWarning(
                "Unknown optimizer, no optimization step was performed"
            )

        flakes_color_img = (
            (1.0 / 2.0)
            * 1.0
            / (1.0 - eps_box)
            * (torch.tanh(flakes_color_inf) + (1 - eps_box))
        )
        weather = (
            initpos + offsets.clone().detach(),
            motion + motion_offsets.clone().detach(),
            flakes,
            flakes_color_img.clone().detach(),
            flakes_transp.clone().detach(),
        )
        with torch.no_grad():
            preds = model(
                image1.detach().clone(),
                image2.detach().clone(),
                weather=weather,
                scene_data=scene_data,
                args_=attack_args,
            )
        flow_weather_pred = preds["flows"].squeeze(0)
        flow_weather_pred = flow_weather_pred.to(device)
        # print("PRINT Pred is: ", flow_weather_pred)
        torch.cuda.empty_cache()

    # save image
    weather_min = (
        initpos + offsets_min,
        motion + motion_offsets_min,
        flakes,
        flakes_color_min,
        flakes_transp_min,
    )
    image1_weather, image2_weather = render(
        image1.detach().clone(),
        image2.detach().clone(),
        scene_data,
        weather_min,
        attack_args,
    )

    # ======== Save Attacked Images ========
    distortion_folder = output_path / "attacked_sintel_dataset" / scene
    distortion_folder.mkdir(parents=True, exist_ok=True)

    save_image(
        image1_weather,
        batch_index,
        distortion_folder,
        image_name=f"frame_{image_index:04d}.png",
        unit_input=True,
        normalize_max=None,
        force_name=True,
    )
    save_image(
        image2_weather,
        batch_index,
        distortion_folder,
        image_name=f"frame_{(image_index + 1):04d}.png",
        unit_input=True,
        normalize_max=None,
        force_name=True,
    )

    # ======== Save Predicted Flow for Origin Image =======
    flow_folder = output_path / "flow_for_origin" / scene
    flow_folder.mkdir(parents=True, exist_ok=True)
    save_flow(
        flow_init,
        batch_index,
        flow_folder,
        flow_name=f"ORIGIN_[im_{image_index}]_[w_step_{attack_args['weather_steps']}]_[w_m_samples_{attack_args['weather_motionblur_samples']}]",
    )

    # ======== Save Predicted Flow for Attacked Image =======
    flow_folder = output_path / "flow_for_attacked" / scene
    flow_folder.mkdir(parents=True, exist_ok=True)
    save_flow(
        flow_weather_pred,
        batch_index,
        flow_folder,
        flow_name=f"ATTACKED_[im_{image_index}]_[w_step_{attack_args['weather_steps']}]_[w_m_samples_{attack_args['weather_motionblur_samples']}]",
    )

    return preds, image1_weather, image2_weather
