# System Architecture

## Component Overview

### 1. Frontend API Service (Cloud Run Service)

- **Purpose:** REST API for job submission and monitoring
- **Resources:** 1 CPU, 512MB RAM
- **Scaling:** 0–10 instances
- **Cost:** $0 when idle (scales to zero)

### 2. Training Worker (Cloud Run Job)

- **Purpose:** GPU-accelerated LoRA fine-tuning
- **Resources:** 8 CPUs, 32GB RAM, 1× L4 GPU
- **Configuration:** europe-west1, no zonal redundancy
- **Cost:** ~$0.67/hour

### 3. Storage Layer (Cloud Storage)

- **Purpose:** Store trained LoRA adapters
- **Structure:**
```bash
gs://lora-training-data-PROJECT_ID/
└── models/
    └── lora_model/
        ├── adapter_model.safetensors (14MB)
        ├── adapter_config.json
        └── tokenizer files
```

## Data Flow

### 1. Job Submission

- User sends POST /train to API Service
- API Service triggers Cloud Run Job
- Returns execution_id to user

### 2. Training

- Cloud Run Job provisions L4 GPU
- Loads pre-cached Gemma model
- Applies LoRA fine-tuning (QLoRA 4-bit)
- Saves adapter weights locally
- Uploads to Cloud Storage

### 3. Monitoring

- User sends GET /status/<id> to API Service
- API queries Cloud Run Executions API
- Returns status (pending/running/completed/failed)

## Optimization Techniques

### QLoRA (4-bit Quantization)

- Base model: 2B params × 16-bit = ~4GB
- With QLoRA: 2B params × 4-bit = ~1GB
- **Savings:** 75% memory reduction

### LoRA (Low-Rank Adaptation)

- Trainable params: 3.6M (0.14% of base model)
- Frozen params: 2.5B (99.86%)
- **Result:** Faster training, smaller checkpoints

### Cloud Run Optimizations

- No zonal redundancy: 35% cost savings
- Pre-cached model in container: Eliminates download time
- Scales to zero: No idle costs