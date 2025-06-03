#!/bin/bash
set -e

mkdir -p datasets
cd datasets
# wget -r -l inf -A png --no-parent --no-check-certificate --reject "index.html*" --cut-dirs=3 -nH https://data.dws.informatik.uni-mannheim.de/machinelearning/robustness_benchmarking/optical_flow/3D_Common_Corruption_Images/