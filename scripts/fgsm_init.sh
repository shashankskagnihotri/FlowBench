models="raft pwcnet gma flownets flownetc flownet2 flowformer rpknet"
datasets="kitti-2015"
checkpoints="things"
targeteds="False"
targets="zero negative"
norms="inf"
attack="fgsm"

for model in $models
do
    for dataset in $datasets
    do
        for norm in $norms
        do
            if [[ $norm = "inf" ]]
            then
                epsilons="8"
                alphas="0.01"
                for epsilon in $epsilons; do
                    epsilon=$(echo "scale=10; $epsilon/255" | bc)
                done
            else
                epsilons="0.251 0.502 0.0005 0.005 0.001 0.05 0.01"
                alphas="0.1 0.2 0.0000001"
            fi
            for targeted in $targeteds
            do
                if [[ targeted = "True" ]]
                then
                    for target in $targets
                    do
                        for alpha in $alphas
                        do
                            for epsilon in $epsilons
                            do                        
                                job_name="${attack}_targeted_${targeted}_target_${target}_l_${norm}_a_${alpha}_eps_${epsilon}"
                                out_dir="slurm/${job_name}.out"
                                err_dir="slurm/${job_name}.err"
                                sbatch -J $job_name \
                                    --output=$out_dir \
                                    --error=$err_dir \
                                    bim_pgd_cospgd_attack.sh \
                                    $model \
                                    $checkpoint \
                                    $dataset \
                                    $attack \
                                    $norm \
                                    $alpha \
                                    $epsilon \
                                    $targeted \
                                    $target
                            done
                        done
                    done
                else
                    for alpha in $alphas
                    do
                        for epsilon in $epsilons
                        do                        
                            job_name="${attack}_its_${iteration}_targeted_${targeted}_l_${norm}_a_${alpha}_eps_${epsilon}"
                            out_dir="slurm/${job_name}.out"
                            err_dir="slurm/${job_name}.err"
                            sbatch -J $job_name \
                                --output=$out_dir \
                                --error=$err_dir \
                                bim_pgd_cospgd_attack.sh \
                                $model \
                                $checkpoint \
                                $dataset \
                                $attack \
                                $norm \
                                $alpha \
                                $epsilon \
                                $targeted \
                                "zero"
                        done
                    done
                fi
            done
        done
    done
done