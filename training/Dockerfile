FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK (includes gsutil)
RUN curl https://sdk.cloud.google.com | bash
ENV PATH="/root/google-cloud-sdk/bin:${PATH}"

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    import os; \
    token = os.getenv('HF_TOKEN'); \
    print('Downloading Gemma model...'); \
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b-it', token=token, trust_remote_code=True); \
    model = AutoModelForCausalLM.from_pretrained('google/gemma-2b-it', token=token, trust_remote_code=True, torch_dtype='auto', low_cpu_mem_usage=True); \
    print('Model cached successfully')"

# Pre-download Alpaca dataset
RUN python -c "from datasets import load_dataset; \
    print('Downloading Alpaca dataset...'); \
    dataset = load_dataset('tatsu-lab/alpaca'); \
    print('Dataset cached successfully')"


COPY train_lora.py .

ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV NCCL_DEBUG=INFO

ENTRYPOINT ["python", "train_lora.py"]
