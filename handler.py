"""
RunPod Serverless handler for Moka models.
Loads GGUF models from network volume on startup, routes by model name.
"""

import os
import runpod
from llama_cpp import Llama

VOLUME_PATH = os.environ.get("MODEL_DIR", "/runpod-volume")
MODELS = {}


def load_models():
    """Load available models from network volume into GPU memory."""
    global MODELS

    moka1_path = os.path.join(VOLUME_PATH, "moka1-Q4_K_M.gguf")
    if os.path.exists(moka1_path):
        print("Loading Moka1 (1.5B Q4_K_M)...")
        MODELS["moka1"] = Llama(
            model_path=moka1_path,
            n_gpu_layers=-1,
            n_ctx=2048,
            verbose=False,
        )

    smart_path = os.path.join(VOLUME_PATH, "moka1-smart-Q4_K_M.gguf")
    if os.path.exists(smart_path):
        print("Loading Moka1-Smart (3B Q4_K_M)...")
        MODELS["moka1-smart"] = Llama(
            model_path=smart_path,
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
