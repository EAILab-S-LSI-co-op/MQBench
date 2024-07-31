CUDA_VISIBLE_DEVICES=1 python3 ptq.py --config=/workspace/MQBench/example/ptq/configs/ptq-adaround.yaml > adaround.out
CUDA_VISIBLE_DEVICES=1 python3 ptq.py --config=/workspace/MQBench/example/ptq/configs/ptq-brecq.yaml > brecq.out
CUDA_VISIBLE_DEVICES=1 python3 ptq.py --config=/workspace/MQBench/example/ptq/configs/ptq-qdrop.yaml > qdrop.out
CUDA_VISIBLE_DEVICES=1 python3 ptq.py --config=/workspace/MQBench/example/ptq/configs/ptq-pact.yaml > pact.out