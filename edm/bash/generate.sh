#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"
nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
port=$(( 47000 + $RANDOM % 1000 ))
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate edm

for dataset in 'cifar10' ; do  # cifar10, cifar100, ffhq, imagenet
  ######################################################################################################
  model="edm"
  top_folder="/home/shared/generative_models/inductive_bias/${model}/${dataset}/training"
#  top_folder="/home/shared/generative_models/training/${model}/${dataset}/self-conditioning"

  S_churn=0  # Disables stochastic sampling
  S_min=0.05
  S_max=50
  S_noise=1.003
  if [ "$dataset" = cifar10 ]; then
      batch=2048
      ref_path="/home/shared/DataSets/cifar-10/cifar10-refs.pkl"
      steps=18
  elif [ "$dataset" = cifar100 ]; then
      batch=2048
      ref_path="/home/shared/DataSets/cifar-100/cifar100-refs.pkl"
      steps=18
  else
      echo "Dataset not supported"
  fi

  #######################################################################################################
  all_labels="/home/shared/generative_models/diffusion_overfit/fd_analysis/${dataset}/train/uniform/subsamples_50000_01/labels.pt"  # from train set

  metrics='none'  # 'none' to disable direct metric computation, otherwise 'fid,fd_dinov2'

  seeds='0-49999'

  net_neg_pkl="None"
  cfg_interval="none"
  snap_pos='100000'
  snap_neg='020009'

  if [[ "$model" == "edm" ]]; then
    rho=7
    train_folder="cond"
  fi

  model_path_pos="${top_folder}/${train_folder}-default"
#  for kmeans in 2 5 10 20 50 100 200 300 400 500 ; do
#    model_path_pos="${top_folder}/kmeans-${kmeans}"
  for method in 'none' ; do  # 'none' or 'regular' for cfg
    # Extract numbers from available snap paths
    extracted_numbers=$(find "$model_path_pos" -maxdepth 1 -type f -name "network-snapshot-*.pkl" | sed -n 's/.*network-snapshot-\([0-9]*\)\.pkl/\1/p' | sort -n)

    # Assign all snaps correctly
    snaps_pos=$(echo "$extracted_numbers")
#    snaps_pos=$(echo "$extracted_numbers" | awk 'NR % 2 == 0')  # every second

    echo "Processing snaps: $(echo "$snaps_pos" | tr '\n' ' ')"
#      for snap_pos in $snaps_pos ; do
    for snap_pos in '200090' ; do
      for cfg_weight in 0 ; do
        net_pos_pkl="${model_path_pos}/network-snapshot-${snap_pos}.pkl"

        if [ "$method" == 'none' ]; then
          cfg_method='none'
          cfg_weight=0
        fi

        if [[ "$method" != 'none' ]]; then
          net_neg_pkl="${top_folder}/${train_folder_neg}/network-snapshot-${snap_neg}.pkl"
        fi

        if [ "$method" == 'regular' ]; then
          cfg_method='regular'
          net_neg_pkl="${top_folder}/uncond-default/network-snapshot-${snap_neg}.pkl"
        fi

        if [ "$kmeans" != 0 ]; then
          outdir="/home/shared/generative_models/diffusion_overfit/fd_analysis/${dataset}/gen/uniform/edm-kmeans${kmeans}/${snap_pos}/${method}/samples/"
        else
          outdir="/home/shared/generative_models/diffusion_overfit/fd_analysis/${dataset}/gen/uniform/edm/${snap_pos}/${method}/samples/"
        fi
        outdir="/home/shared/generative_models/diffusion_overfit/fd_analysis/delete2/"

        echo "Running: $dataset $model $method $cfg_method $cfg_weight $snap_pos $seeds"
        while true; do
          gpu_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
          if [ "$gpu_memory" -gt 35000 ]; then
              echo "Launching experiment..."

              torchrun --master_port=$port --nproc_per_node="$nproc_per_node" generate.py \
                --net_pos_pkl "$net_pos_pkl" \
                --net_neg_pkl "$net_neg_pkl" \
                --outdir "$outdir" \
                --seeds $seeds \
                --subdirs \
                --batch "$batch" \
                --metrics $metrics \
                --ref_path "$ref_path" \
                --steps "$steps" \
                --rho "$rho" \
                --S_churn $S_churn \
                --S_min "$S_min" \
                --S_max "$S_max" \
                --S_noise "$S_noise" \
                --cfg_method "$cfg_method" \
                --cfg_weight "$cfg_weight" \
                --cfg_interval $cfg_interval \
#                --all_labels $all_labels ;
                break
          else
              echo "Waiting for more GPU memory..."
              sleep 5m
          fi
        done  # Close the while loop
      done  # Close the cfg_weight loop
    done  # Close the snap_pos loop
  done  # Close the method loop
done  # Close the dataset loop

# python frechet_utils.py ref --data=/home/shared/DataSets/vision_benchmarks/FFHQ-i/ffhq-64x64.zip --dest=/home/shared/DataSets/vision_benchmarks/FFHQ-i/ffhq64-refs.pkl --batch=256