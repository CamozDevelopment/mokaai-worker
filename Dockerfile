FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir runpod llama-cpp-python

COPY handler.py /handler.py

CMD ["python3", "/handler.py"]
