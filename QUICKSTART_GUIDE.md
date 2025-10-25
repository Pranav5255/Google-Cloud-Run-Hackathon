# Quick Start Guide (For Judges)

## Try It in 2 Minutes!

### Option 1: Web Dashboard (Easiest)
1. Visit the [Deployed Site](https://lora-dashboard-906036652731.europe-west1.run.app/)
2. Click "Submit Training" tab
3. Keep default settings, click "Start Training"
4. Copy the execution ID
5. Go to "Job Status" tab, paste ID, click "Check Status"
6. View trained model in "Models" tab

### Option 2: API (For Developers)
```bash
# Submit training job
curl -X POST https://lora-api-qx2z7yeyna-ew.a.run.app/train \\
  -H "Content-Type: application/json" \\
  -d '{"lora_rank": 16, "num_epochs": 1}'

# Check status (replace EXECUTION_ID)
curl https://lora-api-qx2z7yeyna-ew.a.run.app/status/EXECUTION_ID

# List models
curl https://lora-api-qx2z7yeyna-ew.a.run.app/models
```

## What to Look For

- ✅ **GPU Usage:** Training runs on NVIDIA L4 GPU (check Cloud Run Jobs dashboard)
- ✅ **QLoRA Memory:** Only uses ~5GB VRAM vs. 14GB without quantization
- ✅ **Speed:** Training completes in 2-3 minutes for 100 samples
- ✅ **Cost:** ~\$0.03 per training run
- ✅ **Serverless:** All services scale to zero when idle

## Architecture Highlights

- **3 Cloud Run Services:** Dashboard, API, Training Job
- **Pre-cached Model:** Gemma-2B embedded in container (no download time)
- **Smart Scaling:** No-zonal-redundancy for 35% cost savings
- **Production Ready:** Error handling, monitoring, logging
