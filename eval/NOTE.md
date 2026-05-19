# Launch the AI model for inference

```
vllm serve Qwen/Qwen3.5-4B --port 8000 --gpu-memory-utilization 0.85 --tensor-parallel-size 1 --max-model-len 4000 --reasoning-parser qwen3 --language-model-only
```

# How to run the evals

inspect 

```
OPENAI_API_KEY=EMPTY python -m inspect_ai eval eval/plain_mmlu.py --model vllm/Qwen/Qwen3.5-4B --model-base-url http://localhost:8000/v1
```

