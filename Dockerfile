FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip build-essential cmake && \
    rm -rf /var/lib/apt/lists/*

ENV CMAKE_ARGS="-DGGML_CUDA=on"
RUN pip3 install --no-cache-dir runpod llama-cpp-python

COPY handler.py /handler.py

CMD ["python3", "/handler.py"]
