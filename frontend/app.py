from flask import Flask, request, jsonify
from google.cloud import run_v2
from google.cloud import storage
import os
import uuid

app = Flask(__name__)

PROJECT_ID = os.getenv('PROJECT_ID', 'lora-finetuning-platform')
REGION = 'europe-west1'
JOB_NAME = 'lora-training-job'

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service': 'LoRA Fine-Tuning Platform',
        'version': '1.0',
        'endpoints': {
            '/': 'Service info (this page)',
            '/train': 'POST - Submit training job',
            '/status/<execution_id>': 'GET - Check job status',
            '/models': 'GET - List trained models'
        }
    })

@app.route('/train', methods=['POST'])
def submit_training():
    """Submit a new LoRA training job"""
    try:
        data = request.get_json()
        
        # Get parameters (with defaults)
        model_name = data.get('model_name', 'google/gemma-2b-it')
        lora_rank = data.get('lora_rank', 16)
        num_epochs = data.get('num_epochs', 1)
        learning_rate = data.get('learning_rate', 0.0002)
        
        # Execute Cloud Run Job
        client = run_v2.JobsClient()
        job_path = f"projects/{PROJECT_ID}/locations/{REGION}/jobs/{JOB_NAME}"
        
        operation = client.run_job(name=job_path)
        
        # Get execution name from operation
        execution_name = operation.metadata.name.split('/')[-1]
        
        return jsonify({
            'status': 'success',
            'message': 'Training job submitted',
            'execution_id': execution_name,
            'parameters': {
                'model_name': model_name,
                'lora_rank': lora_rank,
                'num_epochs': num_epochs,
                'learning_rate': learning_rate
            },
            'check_status': f'/status/{execution_name}'
        }), 202
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/status/<execution_id>', methods=['GET'])
def check_status(execution_id):
    """Check the status of a training job"""
    try:
        client = run_v2.ExecutionsClient()
        execution_path = f"projects/{PROJECT_ID}/locations/{REGION}/jobs/{JOB_NAME}/executions/{execution_id}"
        
        execution = client.get_execution(name=execution_path)
        
        # Get status
        status = 'unknown'
        if execution.succeeded_count > 0:
            status = 'completed'
        elif execution.failed_count > 0:
            status = 'failed'
        elif execution.running_count > 0:
            status = 'running'
        else:
            status = 'pending'
        
        return jsonify({
            'execution_id': execution_id,
            'status': status,
            'succeeded_tasks': execution.succeeded_count,
            'failed_tasks': execution.failed_count,
            'running_tasks': execution.running_count,
            'log_uri': f'https://console.cloud.google.com/run/jobs/executions/details/{REGION}/{execution_id}'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 404

@app.route('/models', methods=['GET'])
def list_models():
    """List all trained models in Cloud Storage"""
    try:
        storage_client = storage.Client()
        bucket_name = f'lora-training-data-{PROJECT_ID}'
        bucket = storage_client.bucket(bucket_name)
        
        # List all blobs in models/ directory
        blobs = bucket.list_blobs(prefix='models/')
        
        models = {}
        for blob in blobs:
            # Group by model directory
            parts = blob.name.split('/')
            if len(parts) >= 3:
                model_dir = parts[1]
                if model_dir not in models:
                    models[model_dir] = {
                        'name': model_dir,
                        'files': [],
                        'size_mb': 0
                    }
                models[model_dir]['files'].append(parts[-1])
                models[model_dir]['size_mb'] += blob.size / (1024 * 1024)
        
        return jsonify({
            'models': list(models.values()),
            'total_models': len(models)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
