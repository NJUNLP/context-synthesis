export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export VLLM_NCCL_SO_PATH=/cpfs01/shared/XNLP_H800/software/nccl/build/lib/libnccl.so.2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Generate synthetic context
python generate_synthetic_context.py

# Convert to chat format
python convert_to_chat_format.py