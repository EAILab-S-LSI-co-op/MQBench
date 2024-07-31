CUDA_VISIBLE_DEVICES=1 python3 ptq.py --config=/workspace/MQBench/example/advanced_ptq/configs/ptq-adaround.yaml > adaround.out
CUDA_VISIBLE_DEVICES=1 python3 ptq.py --config=/workspace/MQBench/example/advanced_ptq/configs/ptq-brecq.yaml > brecq.out
CUDA_VISIBLE_DEVICES=1 python3 ptq.py --config=/workspace/MQBench/example/advanced_ptq/configs/ptq-qdrop.yaml > qdrop.out