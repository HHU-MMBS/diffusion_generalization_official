#!/bin/bash

conda activate edm

model=edm2-img64-m

# Reconstruct a new EMA profile with std=0.150
python reconstruct_phema.py \
  --indir="/home/shared/generative_models/inductive_bias/edm2/IN64/models/${model}" \
  --outdir="/home/shared/generative_models/inductive_bias/edm2/IN64/models/${model}" \
  --outprefix "${model}" \
  --outstd=0.175 \
  --outkimg=0268435 ;