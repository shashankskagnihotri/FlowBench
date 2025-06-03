import argparse
import json

from pathlib import Path
from ptlflow import get_model_reference
from typing import Dict, Optional

from .attacks import _init_parser


def get_args_parser(model_name: str, dataset: str) -> argparse.Namespace:
    # merge threat arguments with optical flow model-specific arguments
    parser = _init_parser()
    model_ref = get_model_reference(model_name)
    parser = model_ref.add_model_specific_args(parser)
    args = parser.parse_args()

    # update common arguments
    args.model = model_name
    args.val_dataset = dataset
    args.pretrained_ckpt = get_pretrained_ckpt(dataset)
    args.write_outputs = True
    if dataset == 'kitti-2015':
        args.kitti_2012_root_dir = None
        args.kitti_2015_root_dir = 'datasets/kitti2015'
    elif dataset in ['sintel-clean', 'sintel-final']:
        args.mpi_sintel_root_dir = 'datasets/Sintel'

    return args


def get_pretrained_ckpt(dataset: str) -> str:
    pretrained_ckpts = {
        'kitti-2015': 'kitti',
        'sintel-clean': 'sintel',
        'sintel-final': 'sintel'
    }
    return pretrained_ckpts.get(dataset, dataset)


def get_results(result_path: Path) -> Dict[str, Optional[float]]:
    if not result_path.is_file():
        raise FileNotFoundError(f'JSON file not found: {result_path}.')
    
    with open(result_path,'r') as f:
        data = json.load(f)
    
    metrics = ['epe', 'px3', 'cosim']
    results = {metric: data['experiments'][-1]['metrics'].get(metric) for metric in metrics}
    return results