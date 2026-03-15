FROM python:3.10

ENV PYTHONUNBUFFERED=1

# Install libgomp (required by llama-cpp-python) and cmake for building from source
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 cmake && rm -rf /var/lib/apt/lists/*

# Install runpod SDK
RUN pip install --no-cache-dir runpod

# Build llama-cpp-python from source with generic CPU flags (avoids SIGILL on varied hardware)
ENV CMAKE_ARGS="-DLLAMA_NATIVE=OFF"
RUN pip install --no-cache-dir llama-cpp-python --no-binary llama-cpp-python

COPY handler.py /handler.py

CMD ["python3", "/handler.py"]
