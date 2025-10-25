import streamlit as st
import requests
import json

API_URL = "https://lora-api-qx2z7yeyna-ew.a.run.app"

st.set_page_config(page_title="LoRA Fine-Tuning Platform", page_icon="🚀", layout="wide")

st.title("🚀 LoRA Fine-Tuning Platform")
st.markdown("*Serverless LLM fine-tuning on Google Cloud Run with GPU acceleration*")

# Sidebar
with st.sidebar:
    st.header("📊 Platform Stats")
    try:
        models = requests.get(f"{API_URL}/models").json()
        st.metric("Trained Models", models['total_models'])
        if models['total_models'] > 0:
            st.metric("Total Size", f"{sum([m['size_mb'] for m in models['models']]):.1f} MB")
    except:
        st.warning("Unable to fetch stats")

# Main tabs
tab1, tab2, tab3 = st.tabs(["🎯 Submit Training", "📈 Job Status", "📦 Models"])

with tab1:
    st.header("Submit LoRA Training Job")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_name = st.selectbox(
            "Base Model",
            ["google/gemma-2b-it", "google/gemma-7b-it"],
            index=0
        )
        lora_rank = st.slider("LoRA Rank", 4, 64, 16, help="Higher rank = more capacity but slower")
    
    with col2:
        num_epochs = st.slider("Epochs", 1, 5, 1)
        learning_rate = st.number_input("Learning Rate", value=0.0002, format="%.5f")
    
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        with st.spinner("Submitting job..."):
            payload = {
                "model_name": model_name,
                "lora_rank": lora_rank,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate
            }
            
            try:
                response = requests.post(f"{API_URL}/train", json=payload)
                if response.status_code == 202:
                    result = response.json()
                    st.success(f"✅ Job submitted! Execution ID: `{result['execution_id']}`")
                    st.info(f"Check status in the 'Job Status' tab")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Failed to submit: {str(e)}")

with tab2:
    st.header("Check Job Status")
    
    execution_id = st.text_input("Execution ID", placeholder="lora-training-job-xxxxx")
    
    if st.button("🔍 Check Status", use_container_width=True):
        if execution_id:
            try:
                response = requests.get(f"{API_URL}/status/{execution_id}")
                if response.status_code == 200:
                    status = response.json()
                    
                    status_emoji = {
                        "completed": "✅",
                        "running": "⏳",
                        "failed": "❌",
                        "pending": "⏸️"
                    }
                    
                    st.subheader(f"{status_emoji.get(status['status'], '❓')} Status: {status['status'].upper()}")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Succeeded Tasks", status['succeeded_tasks'])
                    col2.metric("Running Tasks", status['running_tasks'])
                    col3.metric("Failed Tasks", status['failed_tasks'])
                    
                    if status.get('log_uri'):
                        st.markdown(f"[📋 View Logs]({status['log_uri']})")
                else:
                    st.error("Job not found")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter an execution ID")

with tab3:
    st.header("Trained Models")
    
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()
    
    try:
        response = requests.get(f"{API_URL}/models")
        models = response.json()
        
        if models['total_models'] > 0:
            for model in models['models']:
                with st.expander(f"📦 {model['name']} ({model['size_mb']:.1f} MB)"):
                    st.write("**Files:**")
                    for file in model['files']:
                        st.text(f"  • {file}")
                    
                    st.code(f"gs://lora-training-data-lora-finetuning-platform/models/{model['name']}/", language="bash")
        else:
            st.info("No models found. Submit a training job to get started!")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built for **Google Cloud Run Hackathon 2025** | [API Docs](https://lora-api-qx2z7yeyna-ew.a.run.app)")
