import torch
from transformers import Trainer, TrainingArguments, DefaultDataCollator
from peft import LoraConfig, get_peft_model
import re

def apply_lora_and_finetune(
    model, 
    train_dataset, 
    eval_dataset, 
    output_dir="./viswanda_recovery",
    learning_rate=2e-4, 
    num_epochs=3, 
    batch_size=8
):
    print("\n=== Starting Post-Pruning LoRA Recovery ===")

    # 1. Freeze Base Model
    for param in model.parameters():
        param.requires_grad = False

    # 2. Configure LoRA

    # CRITICAL CHANGE: Use Regex to distinguish MLP Output from Attention Output
    # 1. "intermediate.dense" -> The MLP Expansion layer
    # 2. "output.dense" -> The MLP Projection layer (We use regex to ensure it's NOT attention)
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,      # Reduced from 256 to prevent overfitting
        target_modules=[
            "intermediate.dense",
            # This regex matches "layer.0.output.dense" but NOT "attention.output.dense"
            r".*layer\.\d+\.output\.dense$" 
        ],
        lora_dropout=0.1,    # Increased from 0.05 to fight the 0.02 loss
        bias="none",
        modules_to_save=["classifier", "layernorm"] # CRITICAL: Retrain LayerNorms to fix distribution shift
    )

    model = get_peft_model(model, peft_config)
    print("Verifying Trainable Parameters (Expect Adapters + LayerNorms):")
    model.print_trainable_parameters()

    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8, 
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=20,
        save_strategy="no",
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        dataloader_num_workers=0,
        fp16=False,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DefaultDataCollator(),
    )

    print("Training...")
    trainer.train()

    print("Recovery Fine-Tuning Complete.")
    return model