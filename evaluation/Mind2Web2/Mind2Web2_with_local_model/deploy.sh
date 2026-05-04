#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/model/Qwen3-4B-Instruct-2507}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-eval_model}"
 
# load Intel
module load intel/2021.10.0
 
export LD_LIBRARY_PATH=/apps/spack/0.21/cardinal/linux-rhel9-sapphirerapids/intel-oneapi-compilers/gcc/11.3.1/2023.2.3-7rf6exw/compiler/2023.2.3/linux/compiler/lib/intel64_lin:$LD_LIBRARY_PATH
 
 
export CUDA_HOME=/usr/local/cuda
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG=INFO          
# export NCCL_IB_DISABLE=1         
# export VLLM_USE_TORCH_COMPILE=0
 

for i in 0 1 2 3; do
  gpu=$i
  port=$((6000 + i))
  echo "Starting service on GPU $gpu, port $port (single GPU)"
  CUDA_VISIBLE_DEVICES=$gpu python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 --port $port \
    --served-model-name "$SERVED_MODEL_NAME" \
    --model "$MODEL_DIR" \
    --gpu-memory-utilization 0.9   \
    -tp 1  &
  sleep 2  
done
 
 
wait  
