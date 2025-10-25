import os
import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from google.cloud import storage
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning on Cloud Run")
    parser.add_argument("--model_name", type=str, default="google/gemma-2b-it",
                        help="Base model to fine-tune")
    parser.add_argument("--dataset_path", type=str, default="gs://lora-training-data-lora-finetuning-platform/sample_data.jsonl",
                        help="Path to training dataset in GCS")
    parser.add_argument("--output_path", type=str, default="gs://lora-training-data-lora-finetuning-platform/models",
                        help="Output path for fine-tuned model")
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank (r)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha scaling")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get HuggingFace token from environment and strip whitespace
    hf_token = os.getenv("HF_TOKEN", "").strip()
    
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(project="lora-finetuning", config=vars(args))
    
    print("=" * 50)
    print("LoRA Fine-tuning on Cloud Run with GPU")
    print("=" * 50)
    print(f"Model: {args.model_name}")
    print(f"LoRA Rank: {args.lora_rank}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epochs}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"HF Token: {'Present' if hf_token else 'Missing'}")
    print("=" * 50)
    
    # Load tokenizer with authentication - use trust_remote_code for Gemma
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        token=hf_token,
        trust_remote_code=True
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with 4-bit quantization (QLoRA)
    print("\n[2/6] Loading base model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        token=hf_token,
        trust_remote_code=True
    )
    
    # Prepare model for training
    print("\n[3/6] Preparing model for LoRA training...")
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
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
    
    # Training arguments - DISABLE CHECKPOINTING
    print("\n[5/6] Setting up training...")
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=True,
        logging_steps=5,
        save_strategy="no",  # DISABLE AUTOMATIC SAVING
        report_to="wandb" if args.use_wandb else "none",
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
    
    # Save model ONLY at the end - locally
    print("\nSaving model locally...")
    model.save_pretrained("./lora_model", safe_serialization=True)
    tokenizer.save_pretrained("./lora_model")
    
    # Upload to GCS
    print("\nUploading model to Cloud Storage...")
    os.system(f"gsutil -m cp -r ./lora_model {args.output_path}/")
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print(f"Model saved to: {args.output_path}/lora_model")
    print("=" * 50)
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
