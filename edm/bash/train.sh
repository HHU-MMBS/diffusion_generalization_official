# TRAINING

# for our diffusion model
#--loss_fn ours \
#--sigma_dist ours \
#--sigma_eps 5e-2 \
#--sigma_od_min 1e-3 \
#--sigma_od_max 90 \
#--gamma 6 \

# Start of bash code
export CUDA_VISIBLE_DEVICES="0,1,2,3"
nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
port=$(( 47000 + $RANDOM % 1000 ))
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate edm
######################################################################################################
# Set global parameters here that don't change between runs
cond=1  # (un-)conditional
dataset=cifar10
duration=500
loss_fn=karras
######################################################################################################
num_blocks=4
if [[ "$dataset" == cifar10 || "$dataset" == cifar100 ]]; then
    cres=2,2,2    # {num * base_dim}_blocks
    lr=1e-3
    batch=1024
    dropout=0.13
    augment=0.12
elif [[ "$dataset" == ffhq ]]; then
    cres=1,2,2,2
    lr=2e-4
    batch=512
    dropout=0.05
    augment=0.15
elif [[ "$dataset" == afhqv2 ]]; then
    cres=1,2,2,2
    lr=2e-4
    batch=512
    dropout=0.25
    augment=0.15
elif [[ "$dataset" == cifar100-coarse ]]; then
    cres=2,2,2
    lr=1e-3
    batch=1024
    dropout=0.13
    augment=0.12
else
    echo "Dataset not supported"
fi
#######################################################################################################
# Set the other parameters here
wd=0
#wd=2e-3
#outdir="/home/shared/generative_models/inductive_bias/edm/${dataset}/training/uncond-wd${wd}-ft"  # todo always double-check this
#num_blocks=4
#cres=1,1,1
#outdir="/home/shared/generative_models/inductive_bias/edm/${dataset}/training/uncond-cres${cres}-blocks${num_blocks}"  # todo always double-check this

#ra=0.5  # range alpha
#outdir="/home/shared/generative_models/inductive_bias/ours/${dataset}/training/cond-range_weight"  # todo always double-check this

outdir="/home/shared/generative_models/inductive_bias/${loss_fn}/${dataset}/training/cond-default_new"  # todo always double-check this

echo "Running ${dataset} ${duration}M cond=${cond} loss=${loss_fn} cres=${cres} blocks=${num_blocks} wd=${wd}"
while true; do
  gpu_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
  if [ $gpu_memory -gt 35000 ]; then
    echo "Launching experiment..."

    torchrun --master_port=$port --nproc_per_node=$nproc_per_node train.py \
      --outdir $outdir \
      --nosubdir 1 \
      --dataset $dataset \
      --cond $cond \
      --loss_fn $loss_fn \
      --duration $duration \
      --cres $cres \
      --num_blocks $num_blocks \
      --lr $lr \
      --batch $batch \
      --dropout $dropout \
      --augment $augment \
      --weight_decay $wd \
      --wandb 0 \
      --tick 2000 \
      --snap 10 \
      --dump 10 \
      --fp16 1 \
      --seed 1948593099 \
      --sigma_od_min 1e-3 \
      --sigma_od_max 90 \
      --sigma_eps 5e-2 \
      --gamma 6 \
      --sigma_od_min 1e-3 \
      --sigma_od_max 90 \
      --resume "/home/shared/generative_models/inductive_bias/edm/${dataset}/training/cond-default/training-state-500000.pt" \
#      --net_neg_pkl /home/shared/generative_models/inductive_bias/ours/cifar10/training/cond-s-eps5e-2_g6-s1-default/network-snapshot-020009.pkl ;
#        --update_eff_ema 1 \
#        --eff_ema_weight 0.5 \

    chmod -R 777 $outdir
    break  # Exit the loop if condition is met
  else
    echo "Waiting for more GPU memory..."
    sleep 5m
  fi
done

# cifar10 ours:
# Duration 100M, small model, bs 1024: 4 GPUs: 8h, 2 GPUs: 12h, 1 GPU: 18h
# Duration 200M, full model, bs 1024: 4GPUs: 24h, 2 GPUs: 44h, 1 GPU: Out of memory
# Duration 200M, full model, bs 512: 4GPUs: 29h, 2 GPUs: 48h, 1 GPU: 86h
