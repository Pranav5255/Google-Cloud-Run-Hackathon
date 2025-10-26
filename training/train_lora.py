import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import json

def main():
    # Read parameters from environment variables
    model_name = os.getenv('MODEL_NAME', 'google/gemma-2b-it')
    lora_rank = int(os.getenv('LORA_RANK', '16'))
    num_epochs = int(os.getenv('NUM_EPOCHS', '1'))
    learning_rate = float(os.getenv('LEARNING_RATE', '0.0002'))
    hf_token = os.getenv('HF_TOKEN', '').strip()
    
    print("=" * 50)
    print("AdaptML - LoRA Fine-tuning with Custom Parameters")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"LoRA Rank: {lora_rank}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("=" * 50)
    
    # Load tokenizer
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with 4-bit quantization
    print("\n[2/6] Loading base model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        token=hf_token,
        trust_remote_code=True
    )
    
    # Save base model config locally for later
    base_model_config = model.config.to_dict()
    
    # Prepare model for training
    print("\n[3/6] Preparing model for LoRA training...")
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    print(f"\n[3/6] Configuring LoRA with rank={lora_rank}...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print("\n[4/6] Loading dataset...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:100]")
    
    def preprocess_function(examples):
        texts = [f"Instruction: {inst}\n\nInput: {inp}\n\nResponse: {out}" 
                 for inst, inp, out in zip(examples["instruction"], 
                                          examples["input"], 
                                          examples["output"])]
        return tokenizer(texts, truncation=True, max_length=512, padding="max_length")
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    
    # Training arguments
    print(f"\n[5/6] Setting up training with lr={learning_rate}, epochs={num_epochs}...")
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=False,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        optim="paged_adamw_8bit",
        warmup_steps=10,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # Train
    print("\n[6/6] Starting training...")
    trainer.train()
    
    # Save model WITHOUT network calls
    print("\nSaving model locally (offline mode)...")
    os.makedirs("./lora_model", exist_ok=True)
    
    # Save adapter weights directly
    model.save_pretrained("./lora_model", safe_serialization=True)
    
    # Save tokenizer
    tokenizer.save_pretrained("./lora_model")
    
    # Save base model config manually (avoid HF lookup)
    with open("./lora_model/base_model_config.json", "w") as f:
        json.dump(base_model_config, f, indent=2)
    
    # Upload to GCS
    print("\nUploading model to Cloud Storage...")
    project_id = os.getenv('PROJECT_ID', 'lora-finetuning-platform')
    
    # Use timestamp for unique model name
    import time
    timestamp = int(time.time())
    model_path = f"models/lora_model_{timestamp}"
    
    os.system(f"gsutil -m cp -r ./lora_model gs://lora-training-data-{project_id}/{model_path}/")
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print(f"Model saved to: gs://lora-training-data-{project_id}/{model_path}")
    print(f"Parameters used: rank={lora_rank}, lr={learning_rate}, epochs={num_epochs}")
    print("=" * 50)

if __name__ == "__main__":
    main()
