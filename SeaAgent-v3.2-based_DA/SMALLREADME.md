# SeaAgent 启动命令

```bash
pip install -e .
python -m web.app
```
```bash
modelscope download --model Qwen/Qwen3-VL-Embedding-2B --local_dir /home/lab-203/llm_model/Qwen3-VL-Embedding-2B
```
## 启动后端服务

### LLM 推理（Qwen3-VL）

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /home/lab-203/qwen9b \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-AWQ \
  --max-model-len 10240 \
  --port 8890 \
  --gpu-memory-utilization 0.60 \
  --max-num-seqs 10 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
```
### Embedding（Qwen3-Embedding-0.6B）
```bash
python -m vllm.entrypoints.openai.api_server \
  --model "/home/lab-203/llm_model/Qwen3-VL-Embedding-2B" \
  --served-model-name "Qwen/Qwen3-VL-Embedding-2B" \
  --runner pooling \
  --convert embed \
  --trust-remote-code \
  --dtype bfloat16 \
  --api-key abc123 \
  --gpu-memory-utilization 0.30 \
  --max-model-len 2048 \
  --port 8891
```