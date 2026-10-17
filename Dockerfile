# CUDA runtime image for serving. Build: docker build -t tensorline .
# Run:  docker run --gpus all -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface tensorline \
#         --model Qwen/Qwen2.5-1.5B-Instruct --quant int4
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY pyproject.toml README.md ./
COPY tensorline ./tensorline
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir "triton>=3.2,<3.3"
EXPOSE 8000
ENTRYPOINT ["tensorline", "--host", "0.0.0.0"]
