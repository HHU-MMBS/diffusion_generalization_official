#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"
nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
port=$(( 47000 + $RANDOM % 1000 ))
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate edm

while true; do
  gpu_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
  if [ $gpu_memory -gt 30000 ]; then
    echo "Launching experiment..."

  proj_folder="/home/shared/generative_models/diffusion_overfit"
  model_folder="/home/shared/generative_models/EDM2"
  net_neg_pkl="none"

  batch=512
  metrics='none'  # 'none' to disable direct metric computation, otherwise 'fid,fd_dinov2'

  # Model settings
  ema_p="0.100"
  ema_n="0.100"

  # Early inference stop
  early_stops='-1'

  for resolution in 512; do  # 64 or 512
    if [ "$resolution" == '64' ]; then
      model_sizes='xl'  # 'xs s m l xl'
    elif [ "$resolution" == '512' ]; then
      model_sizes='xxl'  # 'xxs xs s m l xl xxl'
    fi

    # Class prior
    all_labels="${proj_folder}/fd_analysis/in${resolution}/val/uniform/subsamples_50000_01/labels.pt"  # from val set
    for method in 'none'; do  # 'none' or 'regular' for cfg
      for model_size in $model_sizes; do
        echo $model_size

        for early_stop in $early_stops; do
          model_path="${model_folder}/IN${resolution}/models/edm2-img${resolution}-${model_size}"
  #        model_path="https://nvlabs-fi-cdn.nvidia.com/edm2/raw-snapshots/edm2-img${resolution}-${model_size}"

          # Extract numbers from available snap paths
          escaped_ema_p=$(echo "$ema_p" | sed 's/\./\\./g')
          extracted_numbers=$(find "$model_path" -maxdepth 1 -type f -name "*${escaped_ema_p}*" | awk -F'[-.]' '{print $(NF-3)}' | sort -n)

#          snaps_pos=$(echo "$extracted_numbers" | awk "NR % ${snap_steps} == 0")  # every k-th snap
          snaps_pos='1073741'  # 1073741'  # '2147483'

          # Print the result
          echo "Processing snaps: $(echo "$snaps_pos" | tr '\n' ' ')"
          for snap_pos in $snaps_pos  ; do  # "0067108" "0268435" "2147483"
            for g_weight in 0.0; do
              for g_interval in "none" ; do  # "15-20" "17-20"
                for seeds in "0-49999" ; do
                  # Load model locally or from NVIDIA via url
                  net_pos_pkl="${model_path}/edm2-img${resolution}-${model_size}-${snap_pos}-${ema_p}.pkl"

                  # Where to save the samples
                  outdir="${proj_folder}/fd_analysis/in${resolution}/gen/uniform/edm2/${model_size}/${snap_pos}/${method}/samples/"
                  if [[ "$method" != 'none' && "$early_stop" != -1 ]]; then
                    outdir="${proj_folder}/fd_analysis/in${resolution}/gen/uniform/edm2/${model_size}/${snap_pos}/${method}/w${g_weight}_stop${early_stop}/samples/"
                  elif [ "$early_stop" != -1 ]; then
                    outdir="${proj_folder}/fd_analysis/in${resolution}/gen/uniform/edm2/${model_size}/${snap_pos}/${method}/stop${early_stop}/samples/"
                  else
                    outdir="${proj_folder}/fd_analysis/in${resolution}/gen/uniform/edm2/${model_size}/${snap_pos}/${method}/w${g_weight}/samples/"
                  fi

                  outdir="${proj_folder}/fd_analysis/delete5/"

                  # Skip to the next snapshot if the output directory already exists
                  if [ -d "$outdir" ]; then
                    echo "Output directory $outdir already exists. Skipping to the next run..."
                    continue
                  fi

                  # Set the right guidance method
                  snap_neg=$snap_pos
                  if [ "$method" == 'auto' ]; then
                    g_method='wmg'
                    snap_neg='0268435'  # "0268435" 0134217
                    neg_model_size="xs"
                  elif [ "$method" == 'regular' ]; then
                    g_method='regular'
                    neg_model_size="xs-uncond"
                  else
                    g_method='none'
                    g_weight=0
                  fi

                  # Load a negative model if needed for guidance
                  if [[ "$g_method" != 'none' ]]; then
                    net_neg_pkl="https://nvlabs-fi-cdn.nvidia.com/edm2/raw-snapshots/edm2-img${resolution}-${neg_model_size}/edm2-img${resolution}-${neg_model_size}-${snap_neg}-${ema_n}.pkl"
                    net_neg_pkl="${model_folder}/IN${resolution}/models/edm2-img${resolution}-${neg_model_size}/edm2-img${resolution}-${neg_model_size}-${snap_neg}-${ema_n}.pkl"
                  else
                    net_neg_pkl='none'
                  fi

                  echo "Running with g_weight $g_weight, g_method $g_method, seeds $seeds, ema_p $ema_p, ema_n $ema_n, method $method, g_interval $g_interval", early_stop $early_stop
                  echo "Positive network: " $net_pos_pkl
                  echo "Negative network: " $net_neg_pkl

                  torchrun --master_port=$port --nproc_per_node="$nproc_per_node" generate_images.py \
                    --net_pos_pkl "$net_pos_pkl" \
                    --net_neg_pkl "$net_neg_pkl" \
                    --outdir "$outdir" \
                    --seeds $seeds \
                    --subdirs \
                    --batch "$batch" \
                    --metrics $metrics \
                    --ref_path "https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img${resolution}.pkl" \
                    --all_labels $all_labels \
                    --g_weight $g_weight \
                    --g_method $g_method \
                    --g_interval $g_interval \
                    --early_stop "$early_stop" ;
                done
              done
            done
          done
        done
      done
    done
  done
  break
  else
      echo "Waiting for more GPU memory..."
      sleep 15m
  fi
done

# python dataset_tool.py convert --source=/home/shared/DataSets/vision_benchmarks/imagenet-kaggle/ILSVRC/Data/CLS-LOC/train --dest=/home/shared/DataSets/vision_benchmarks/imagenet-kaggle/img512.zip --resolution=512x512 --transform=center-crop-dhariwal