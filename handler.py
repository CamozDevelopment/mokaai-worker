"""
RunPod Serverless handler for Moka models.
Downloads GGUF models on cold start, caches locally.
"""

import os
import urllib.request
import runpod
from llama_cpp import Llama

MODEL_DIR = "/tmp/models"
MODELS = {}

MODEL_URLS = {
    "moka1": os.environ.get(
        "MOKA1_URL",
        "https://huggingface.co/CamozDevelopment/moka1-gguf/resolve/main/moka1-Q4_K_M.gguf",
    ),
}


def download_model(name, url, dest):
    """Download a model file if not already cached."""
    if os.path.exists(dest):
        print(f"{name} already cached at {dest}")
        return
    print(f"Downloading {name} from {url}...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"{name} downloaded: {size_mb:.0f} MB")


def load_models():
    """Download and load models into GPU memory."""
    global MODELS

    for name, url in MODEL_URLS.items():
        dest = os.path.join(MODEL_DIR, f"{name}.gguf")
        download_model(name, url, dest)
        print(f"Loading {name}...")
        MODELS[name] = Llama(
            model_path=dest,
            n_gpu_layers=-1,
            n_ctx=2048,
            verbose=False,
        )

    print(f"Loaded models: {list(MODELS.keys())}")


def handler(job):
    """
    RunPod handler. Expects input matching OpenAI chat completions format:
    {
        "model": "moka1" | "moka1-smart",
        "messages": [{"role": "...", "content": "..."}],
        "max_tokens": 2048,
        "temperature": 0.7,
        "stream": false  (streaming handled differently on RunPod)
    }
    """
    inp = job["input"]

    model_name = inp.get("model", "moka1")
    if model_name not in MODELS:
        return {"error": f"Unknown model: {model_name}. Available: moka1, moka1-smart"}

    llm = MODELS[model_name]
    messages = inp.get("messages", [])
    max_tokens = min(inp.get("max_tokens", 2048), 4096)
    temperature = inp.get("temperature", 0.7)

    result = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )

    # Return in OpenAI-compatible format
    return {
        "id": result.get("id", ""),
        "object": "chat.completion",
        "model": model_name,
        "choices": result.get("choices", []),
        "usage": result.get("usage", {}),
    }


# Generator handler for streaming support
def stream_handler(job):
    """Streaming handler — yields tokens one at a time."""
    inp = job["input"]

    model_name = inp.get("model", "moka1")
    if model_name not in MODELS:
        yield {"error": f"Unknown model: {model_name}"}
        return

    llm = MODELS[model_name]
    messages = inp.get("messages", [])
    max_tokens = min(inp.get("max_tokens", 2048), 4096)
    temperature = inp.get("temperature", 0.7)

    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        token = delta.get("content", "")
        if token:
            yield token


load_models()

runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,
})
