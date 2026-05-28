NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "Error: No GPU detected. Exiting."
    exit 1
fi
echo "Detected GPUs: ${NUM_GPUS}"

# === NCCL settings: only for multi-GPU ===
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Multi-GPU mode — enabling NCCL optimizations"
    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_IB_HCA=mlx5_2,mlx5_3
    export NCCL_BLOCKING_WAIT=1
    export NCCL_ASYNC_ERROR_HANDLING=1
    export NCCL_TIMEOUT=10000
    export NCCL_SOCKET_TIMEOUT_MS=360000
else
    echo "Single-GPU mode — skipping NCCL env vars"
fi

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenPI
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3.5-0.8B
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
# Framework_name=CosmoPredict2GR00T
# freeze_module_list=''
# base_wm=playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World
# config_yaml=./starVLA/config/training/starvla_test.yaml
# dataset_root=/mnt/pfs/dengyiqi/datasets
# data_mix=x2w_wm_dataset
run_root_dir=./playground/Checkpoints
run_id=0505_test-dataset_CosmoPredict2GR00T
# === End of environment variable configuration ===
###########################################################################################


export WANDB_MODE=offline
# export WANDB_API_KEY="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"

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


num_processes=${NUM_PROCESSES:-$(nvidia-smi -L | wc -l)}

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.world_model.base_wm ${base_wm} \
  --datasets.vla_data.data_root_dir ${dataset_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 20000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero \
  --wandb_entity jinhuiye \
  # --is_debug True



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
