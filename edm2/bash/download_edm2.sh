conda activate edm

model_path='/home/shared/generative_models/inductive_bias'
model='edm2'  # 'edm' or 'edm2'

for res in 64 512 ; do
    for model_size in xs-uncond s m l xl ; do
        mkdir -p "${model_path}/${model}/IN${res}/models/${model}-img${res}-${model_size}"
        rclone copy --progress --http-url "https://nvlabs-fi-cdn.nvidia.com/${model}" ":http:raw-snapshots/${model}-img${res}-${model_size}/" "/home/shared/generative_models/inductive_bias/${model}/IN${res}/models/${model}-img${res}-${model_size}"
    done
done

#rclone copy --progress --http-url https://nvlabs-fi-cdn.nvidia.com/edm2 :http:raw-snapshots/edm2-img512-xs/ /home/shared/generative_models/inductive_bias/edm2/IN512/models/edm2-img${res}-${model_size}