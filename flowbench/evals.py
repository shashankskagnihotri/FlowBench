import os
import pandas as pd
import math

from ptlflow import get_model, get_model_reference
from ptlflow.utils.utils import get_list_of_available_models_list
from pathlib import Path

from .attacks import attack
from .utils import get_args_parser, get_pretrained_ckpt, get_results


common_corruptions_2d = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]

common_corruptions_3d = [
    'far_focus', 'near_focus', 'fog_3d', 'color_quant',
    'iso_noise', 'low_light', 'xy_motion_blur', 'z_motion_blur',
]

attacks = ['fgsm', 'bim', 'pgd', 'cospgd', 'apgd', 'fab', 'pcfa']


def load_model(model_name: str, dataset: str):
    """
    Load a pre-configured optical flow estimation model with pretrained weights.

    Args:
        model_name (str): Name of the optical flow model (e.g., CCMR, RAFT).
        dataset (str): Name of the dataset the model was trained on (e.g., KITTI-2015).
    """
    model_name = model_name.lower()
    if model_name not in get_list_of_available_models_list():
        raise ValueError(f"Model '{model_name}' is not recognized.")

    dataset = dataset.lower()
    if dataset == 'kitti2015':
        dataset = 'kitti-2015'
    pretrained_ckpt = get_pretrained_ckpt(dataset)

    model_ref = get_model_reference(model_name)
    checkpoints = model_ref.pretrained_checkpoints.keys()

    if pretrained_ckpt in checkpoints:
        model = get_model(model_name, pretrained_ckpt)
    else:
        raise ValueError(f"No pretrained weights for model '{model_name}' on dataset '{pretrained_ckpt}'.")
    return model


def evaluate(
    model_name: str, dataset: str,
    retrieve_existing: bool = True, threat_model: str = None,
    iterations: int = 20, epsilon: float = 8/255, alpha: float = 0.01,
    lp_norm: str = 'Linf', targeted: bool = True, target: str = 'zero', optim_wrt: str = 'ground_truth',
    weather: str = 'snow', num_particles: int = 10000,
    severity: int = 3,
    # WeatherConfig parameters
    attack_loss: str = "epe",
    weather_scene_scale: float = 1.0,
    weather_data: str = "datasets/adv_weather_data/weather_particles_red",
    weather_model_iters: int = 32,
    weather_steps: int = 750,
    weather_learn_offset: bool = True,
    weather_learn_motionoffset: bool = True,
    weather_learn_color: bool = True,
    weather_learn_transparency: bool = True,
    weather_optimizer: str = "Adam",
    weather_alph_motion: float = 1000.0,
    weather_alph_motionoffset: float = 1000.0,
    weather_motionblur_samples: int = 10,
    weather_do_motionblur: bool = True,
    weather_motionblur_scale: float = 0.025,
    weather_depth_check: bool = False,
    weather_depth_check_differentiable: bool = False,
    weather_transparency_scale: float = 1.0,
    weather_rendering_method: str = "additive",
    weather_recolor: bool = False,
    weather_flake_r: float = 255.0,
    weather_flake_g: float = 255.0,
    weather_flake_b: float = 255.0,
):
    """
    Evaluate the robustness of optical flow estimation under various conditions, including
    adversarial attacks, adversarial weather, 2D common corruptions, and 3D common corruptions.

    Args:
        model_name (str): Name of the optical flow model (e.g., 'CCMR', 'RAFT').
        dataset (str): Dataset the model was trained on (e.g., 'KITTI-2015').
        retrieve_existing (bool): If True, returns cached results if available; otherwise, reruns the evaluation.
        threat_model (str): Type of perturbation to apply. None indicates clean/unperturbed images.
        iterations (int): Number of attack iterations (used in adversarial attack and adversarial weather).
        epsilon (float): Perturbation budget ε (used in adversarial attack).
        alpha (float): Attack step size ϑ (used in adversarial attack).
        lp_norm (str): Norm used to constrain perturbations (used in adversarial attack).
        targeted (bool): Whether the attack is targeted (used in adversarial attack and adversarial weather).
        target (str): Target flow for a targeted attack (used in adversarial attack and adversarial weather).
        optim_wrt (str): Flow reference for optimization (used in adversarial attack).
        weather (str): Weather type for adversarial weather attack(used in adversarial weather).
        num_particles (int): Number of particles per frame (used in adversarial weather).
        severity (int): Corruption severity level (1–5) used in 2D and 3D common corruptions.
        attack_loss (str): Set the name of the used loss function (mse, epe, cosim).
        weather_scene_scale (float): A global scaling to the scene depth.
        weather_data (str): Path to dataset that contains weather data.
        weather_model_iters (int): Number of iters for gma/raft model.
        weather_steps (int): Number of optimization steps per image.
        weather_learn_offset (bool): Whether to optimize initial position of particles.
        weather_learn_motionoffset (bool): Whether to optimize endpoint of particle motion.
        weather_learn_color (bool): Whether to optimize color of particles.
        weather_learn_transparency (bool): Whether to optimize transparency of particles.
        weather_optimizer (str): Optimizer used for perturbations.
        weather_alph_motion (float): Weighting for the motion loss.
        weather_alph_motionoffset (float): Weighting for the motion offset loss.
        weather_motionblur_samples (int): Number of flakes drawn per blurred flake.
        weather_do_motionblur (bool): Control if particles are rendered with motion blur.
        weather_motionblur_scale (float): Scaling factor for motion blur.
        weather_depth_check (bool): Whether particles are rendered if behind an object.
        weather_depth_check_differentiable (bool): Whether rendering check for occlusion is in compute graph.
        weather_transparency_scale (float): Scaling factor for particle transparency.
        weather_rendering_method (str): Method for rendering particle color.
        weather_recolor (bool): Whether all weather is recolored with given RGB value.
        weather_flake_r (float): R value for particle RGB.
        weather_flake_g (float): G value for particle RGB.
        weather_flake_b (float): B value for particle RGB.
    """
    model_name = model_name.lower()
    if model_name not in get_list_of_available_models_list():
        raise ValueError(f"Model '{model_name}' is not recognized.")

    dataset = dataset.lower()
    if dataset == 'kitti2015':
        dataset = 'kitti-2015'

    model = load_model(model_name, dataset)
    results = {}

    if threat_model:
        threat_model = threat_model.lower()

    if retrieve_existing:
        csv_path = 'eval_csv/evals.csv'
        df = pd.read_csv(csv_path)

    if threat_model is None:
        if retrieve_existing:
            filtered = df[
                (df['model'] == model_name) &
                (df['dataset'] == dataset) &
                (df['checkpoint'] == get_pretrained_ckpt(dataset)) &
                (df['attack'] == 'none')
            ]
            if not filtered.empty:
                metrics = ['epe', 'px3', 'cosim']
                results = {metric: float(filtered.iloc[0][metric]) for metric in metrics}
                return model, results

        # run on unperturbed and unaltered images
        args = get_args_parser(model_name, dataset)
        args.output_path = Path(f'outputs/validate/{model_name}_{args.pretrained_ckpt}/unperturbed')
        args.output_path.mkdir(parents=True, exist_ok=True)

        model = get_model(model_name, args.pretrained_ckpt, args)
        attack(args, model)

        # load results from json
        results = get_results(args.output_path / f'metrics_{dataset}.json')
        return model, results

    elif threat_model in attacks:
        if lp_norm not in ['Linf', 'L2']:
            raise ValueError(f"Invalid lp_norm: {lp_norm}. Must be 'Linf' or 'L2'.")
        if targeted and target not in ['negative', 'zero']:
            raise ValueError(f"Invalid target: {target}. Must be 'negative' or 'zero' when targeted=True.")
        if optim_wrt not in ['ground_truth', 'initial_flow']:
            raise ValueError(f"Invalid optim_wrt: {optim_wrt}. Must be 'ground_truth' or 'initial_flow'.")

        lp_norm = 'inf' if lp_norm == 'Linf' else 'two'

        if retrieve_existing:
            filtered = df[
                (df['model'] == model_name) &
                (df['dataset'] == dataset) &
                (df['checkpoint'] == get_pretrained_ckpt(dataset)) &
                (df['attack'] == threat_model) &
                (df['norm'] == lp_norm) &
                (df['epsilon'].round(4) == round(epsilon, 4)) &
                (df['iterations'] == iterations) &
                (df['alpha'] == alpha) &
                (df['targeted'] == targeted) &
                (df['optim'] == optim_wrt)
            ]
            if not filtered.empty:
                if targeted:
                    filtered = filtered[filtered['target'] == target]

                metrics = ['epe', 'px3', 'cosim']
                results = {metric: float(filtered.iloc[0][metric]) for metric in metrics}
                return model, results

        # run Adversarial Attacks
        args = get_args_parser(model_name, dataset)
        args.output_path = Path(f'outputs/validate/{model_name}_{args.pretrained_ckpt}/{threat_model}')
        args.output_path.mkdir(parents=True, exist_ok=True)
        args.attack = threat_model
        args.attack_optim_target = optim_wrt
        args.attack_norm = lp_norm
        args.attack_epsilon = epsilon
        args.attack_iterations = iterations
        args.attack_alpha = alpha
        args.attack_targeted = targeted
        args.attack_target = target

        model = get_model(model_name, args.pretrained_ckpt, args)
        attack(args, model)

        # load results from json
        results = get_results(args.output_path / f'metrics_{dataset}.json')
        return model, results

    elif threat_model == 'adversarial_weather':
        if retrieve_existing:
            filtered = df[
                (df['model'] == model_name) &
                (df['dataset'] == dataset) &
                (df['checkpoint'] == get_pretrained_ckpt(dataset)) &
                (df['attack'] == 'weather') &
                (df['weather_type'] == weather) &
                (df['targeted'] == targeted)
            ]
            if not filtered.empty:
                if targeted:
                    filtered = filtered[filtered['target'] == target]

                metrics = ['epe', 'px3', 'cosim']
                results = {metric: float(filtered.iloc[0][metric]) for metric in metrics}
                return model, results

        # run Adversarial Weather
        args = get_args_parser(model_name, dataset)
        args.output_path = Path(f'outputs/validate/{model_name}_{args.pretrained_ckpt}/weather/{weather}')
        args.output_path.mkdir(parents=True, exist_ok=True)
        args.attack = 'weather'
        args.attack_targeted = targeted
        args.attack_target = target

        import pudb

        pudb.set_trace()
        model = get_model(model_name, args.pretrained_ckpt, args)
        attack(args, model)

        # load results from json
        results = get_results(args.output_path / f'metrics_{dataset}.json')
        return model, results

    elif threat_model == '2dcommoncorruption':
        if severity < 1 or severity > 5:
            raise ValueError("Severity must be an integer between 1 and 5.")

        if retrieve_existing:
            filtered = df[
                (df['model'] == model_name) &
                (df['dataset'] == dataset) &
                (df['checkpoint'] == get_pretrained_ckpt(dataset)) &
                (df['attack'] == 'common_corruptions') &
                (df['severity'] == severity)
            ]
            if not filtered.empty:
                for corruption in common_corruptions_2d:
                    metrics = ['epe', 'px3', 'cosim']
                    row = filtered[filtered['name'] == corruption]
                    metric_values = {metric: float(row.iloc[0][metric]) for metric in metrics}
                    results[corruption] = metric_values
                return model, results

        # run 2D Common Corruption
        for corruption in common_corruptions_2d:
            args = get_args_parser(model_name, dataset)
            args.output_path = Path(f'outputs/validate/{model_name}_{args.pretrained_ckpt}/2dcc/{corruption}')
            args.output_path.mkdir(parents=True, exist_ok=True)
            args.attack = 'common_corruptions'
            args.cc_name = corruption
            args.cc_severity = severity
            model = get_model(model_name, args.pretrained_ckpt, args)
            attack(args, model)

            # load results from json
            metric_values = get_results(args.output_path / f'metrics_{dataset}.json')
            results[corruption] = metric_values
        return model, results

    elif threat_model == '3dcommoncorruption':
        if severity < 1 or severity > 5:
            raise ValueError("Severity must be an integer between 1 and 5.")

        if retrieve_existing:
            filtered = df[
                (df['model'] == model_name) &
                (df['dataset'] == dataset) &
                (df['checkpoint'] == get_pretrained_ckpt(dataset)) &
                (df['attack'] == '3dcc') &
                (df['3dcc_intensity'] == severity)
            ]
            if not filtered.empty:
                for corruption in common_corruptions_3d:
                    metrics = ['epe', 'px3', 'cosim']
                    row = filtered[filtered['3dcc_corruption'] == corruption]
                    metric_values = {metric: float(row.iloc[0][metric]) for metric in metrics}
                    results[corruption] = metric_values
                return model, results

        # run 3D Common Corruption
        for corruption in common_corruptions_3d:
            args = get_args_parser(model_name, dataset)
            args.output_path = Path(f'outputs/validate/{model_name}_{args.pretrained_ckpt}/3dcc/{corruption}')
            args.output_path.mkdir(parents=True, exist_ok=True)
            args.attack = '3dcc'
            args.tdcc_corruption = corruption
            args.tdcc_intensity = severity
            if dataset == 'kitti-2015':
                args.kitti_2012_3DCC_root_dir = None
                args.kitti_2015_3DCC_root_dir = 'datasets/3D_Common_Corruption_Images/kitti2015'
            elif dataset in ['sintel-clean', 'sintel-final']:
                args.mpi_sintel_3DCC_root_dir = 'datasets/3D_Common_Corruption_Images/Sintel'

            model = get_model(model_name, args.pretrained_ckpt, args)
            attack(args, model)

            # load results from json
            metric_values = get_results(args.output_path / f'metrics_{dataset}.json')
            results[corruption] = metric_values
        return model, results

    else:
        ValueError(f'Threat model {threat_model} is not supported')

    return model, results


if __name__ == "__main__":
    model = load_model(
        model_name='RAFT',
        dataset='KITTI2015',
    )

    model, results = evaluate(
        model_name='RAFT',
        dataset='KITTI2015',
        retrieve_existing=True,
    )
    print(results)

    model, results = evaluate(
        model_name='RAFT',
        dataset='KITTI2015',
        retrieve_existing=True,
        threat_model='PGD',
        iterations=20, epsilon=8/255, alpha=0.01, lp_norm='Linf',
        targeted=True, target='zero', optim_wrt='ground_truth',
    )
    print(results)

    model, results = evaluate(
        model_name='RAFT',
        dataset='Sintel-Final',
        retrieve_existing=True,
        threat_model='Adversarial_Weather',
        weather='snow', num_particles=10000, targeted=True, target='zero',
    )
    print(results)

    model, results = evaluate(
        model_name='RAFT',
        dataset='KITTI2015',
        retrieve_existing=True,
        threat_model='2DCommonCorruption',
        severity=3,
    )
    print(results)

    model, results = evaluate(
        model_name='RAFT',
        dataset='KITTI2015',
        retrieve_existing=True,
        threat_model='3DCommonCorruption',
        severity=3,
    )
    print(results)
