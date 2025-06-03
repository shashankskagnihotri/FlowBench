#!/bin/bash
cd "$(dirname "$0")"
sbatch bim_pgd_cospgd_i3_i10.sh
sbatch bim_pgd_cospgd_i20.sh
 