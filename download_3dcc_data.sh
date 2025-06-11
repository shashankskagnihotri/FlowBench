#!/bin/bash
set -e

mkdir -p datasets
cd datasets
wget https://data.dws.informatik.uni-mannheim.de/machinelearning/robustness_benchmarking/optical_flow/zip_files/3D_Common_Corruption_Image.tar
tar -xvf 3D_Common_Corruption_Image.tar --checkpoint=1000 --checkpoint-action=echo="%T"
