# !pip install accelerate -U
# !pip install matplotlib datasets transformers wandb

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import matplotlib.pyplot as plt
import wandb
import argparse
from huggingface_hub import HfFolder, Repository, login
import os

def split_train_test(dataset, train_ratio=0.9):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])

def configure_trainer_instance(model, tokenizer, train_dataset, test_dataset):
    training_args = TrainingArguments(
        output_dir="./output",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=200,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=256,
        bf16=True,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    return trainer

def plot_training_metrics(trainer, metric_name):
    metric_values = [log[metric_name] for log in trainer.state.log_history if metric_name in log]
    plt.plot(metric_values, label=metric_name)
    plt.xlabel("Training Steps")
    plt.ylabel(metric_name.capitalize())
    plt.title(f"{metric_name.capitalize()} vs Training Steps")
    plt.legend()
    plt.show()

def evaluate_gpt2_model(trainer):
    return trainer.evaluate()

def create_gpt2_model():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir="/workspace/cache")
    # Ensure all special tokens are set
    special_tokens_dict = {
        'bos_token': tokenizer.bos_token,
        'eos_token': tokenizer.eos_token,
        'pad_token': tokenizer.eos_token,
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=2048,
        n_embd=2048,
        n_layer=16,
        n_head=16,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))
    model.init_weights()
    return tokenizer, model

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="longest", max_length=2048)

def print_model_info(model, tokenizer):
    print("\n===== Model Information =====")
    print(f"Model Type: {type(model).__name__}")
    print(f"Number of Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print("\n--- Model Architecture ---")
    print(model)
    
    print("\n--- Model Config ---")
    for key, value in model.config.to_dict().items():
        print(f"{key}: {value}")
    
    print(f"\nVocabulary Size: {len(tokenizer)}")
    print(f"Model Max Length: {model.config.n_positions}")
    print(f"Embedding Size: {model.config.n_embd}")
    print(f"Number of Layers: {model.config.n_layer}")
    print(f"Number of Attention Heads: {model.config.n_head}")

    print("\n--- Special Tokens ---")
    print(f"BOS Token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"EOS Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"PAD Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    print("\n===== End of Model Information =====\n")

def push_to_hub(model, tokenizer, repo_name):
    print(f"Pushing model and tokenizer to Hub repository: {repo_name}")
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    print("Model and tokenizer successfully pushed to Hub!")

login(token="HF_TOKEN_HERE")

# Initialize wandb
wandb.init(project="gpt2-training", name="gpt2-llama3-tokenizer")

# Load GPT-2 model and tokenizer
tokenizer, model = create_gpt2_model()

# Print model information
print_model_info(model, tokenizer)

# Load and tokenize the dataset
dataset = load_dataset("dustinwloring1988/fineweb-edu-test-small-sample", split="train", cache_dir="/workspace/cache")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# Train-test split
train_dataset, test_dataset = split_train_test(tokenized_dataset)

# Configure Trainer instance
trainer = configure_trainer_instance(model, tokenizer, train_dataset, test_dataset)

# Train the model
trainer.train()

# Evaluate the model
eval_results = evaluate_gpt2_model(trainer)
print("Evaluation Results:")
print(eval_results)

# Log final evaluation results to wandb
wandb.log({"final_eval": eval_results})

# Visualize the model performance
plot_training_metrics(trainer, "loss")

# Save the model locally
model.save_pretrained("./output/final_model")
tokenizer.save_pretrained("./output/final_model")

# Push to Hub if flag is set
push_to_hub(model, tokenizer, "dustinwloring1988/fineweb-edu-new-test-delete-2")

# Close wandb run
wandb.finish()
