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
    model_name = os.getenv('MODEL_NAME', 'google/gemma-2b-it')
    lora_rank = int(os.getenv('LORA_RANK', '16'))
    num_epochs = int(os.getenv('NUM_EPOCHS', '1'))
    learning_rate = float(os.getenv('LEARNING_RATE', '0.0002'))
    hf_token = os.getenv('HF_TOKEN', '').strip()
    
    print("=" * 50)
    print("AdaptML - LoRA Fine-tuning")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"LoRA Rank: {lora_rank}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"GPU: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 50)
    
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("\n[2/6] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16,
        load_in_4bit=True, token=hf_token, trust_remote_code=True
    )
    base_model_config = model.config.to_dict()
    
    print("\n[3/6] Preparing for LoRA...")
    model = prepare_model_for_kbit_training(model)
    
    print(f"\n[3.5/6] Configuring LoRA (rank={lora_rank})...")
    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("\n[4/6] Loading dataset...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:100]")
    
    def preprocess_function(examples):
        texts = [f"Instruction: {inst}\n\nInput: {inp}\n\nResponse: {out}" 
                 for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"])]
        return tokenizer(texts, truncation=True, max_length=512, padding="max_length")
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    
    print(f"\n[5/6] Training setup (lr={learning_rate}, epochs={num_epochs})...")
    training_args = TrainingArguments(
        output_dir="./results", num_train_epochs=num_epochs,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=learning_rate, fp16=False, bf16=True,
        logging_steps=5, save_strategy="no", report_to="none",
        optim="paged_adamw_8bit", warmup_steps=10,
    )
    
    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    print("\n[6/6] Training...")
    trainer.train()
    
    # MANUALLY EXTRACT ONLY LORA WEIGHTS - NO PEFT FUNCTIONS!
    print("\n🔒 Extracting adapter weights (offline mode)...")
    os.makedirs("./lora_model", exist_ok=True)
    
    # Get full state dict and filter for LoRA params only
    full_state_dict = model.state_dict()
    adapter_state_dict = {k: v for k, v in full_state_dict.items() if 'lora_' in k}
    
    # Save only adapter weights
    torch.save(adapter_state_dict, "./lora_model/adapter_model.bin")
    
    # Calculate size
    adapter_size_mb = sum(p.numel() * p.element_size() for p in adapter_state_dict.values()) / 1024 / 1024
    print(f"✅ Adapter size: {adapter_size_mb:.2f} MB ({len(adapter_state_dict)} tensors)")
    
    # Save configs
    lora_config.save_pretrained("./lora_model")
    with open("./lora_model/base_model_config.json", "w") as f:
        json.dump(base_model_config, f, indent=2)
    tokenizer.save_pretrained("./lora_model")
    
    print("📤 Uploading to GCS...")
    project_id = os.getenv('PROJECT_ID', 'lora-finetuning-platform')
    import time
    timestamp = int(time.time())
    model_path = f"models/lora_model_{timestamp}"
    os.system(f"gsutil -m cp -r ./lora_model gs://lora-training-data-{project_id}/{model_path}/")
    
    print("\n" + "=" * 50)
    print("✅ Training completed!")
    print(f"📦 gs://lora-training-data-{project_id}/{model_path}")
    print(f"💾 Adapter: {adapter_size_mb:.2f} MB")
    print(f"⚙️  rank={lora_rank}, lr={learning_rate}, epochs={num_epochs}")
    print("=" * 50)

if __name__ == "__main__":
    main()