import torch
from transformers import Trainer, TrainingArguments, DefaultDataCollator
from peft import LoraConfig, get_peft_model

def apply_lora_and_finetune(
    model, 
    train_dataset, 
    eval_dataset, 
    output_dir="./viswanda_recovery",
    learning_rate=1e-4,  # Lower LR for stability
    num_epochs=5,        # Increase epochs (16 steps is not enough)
    batch_size=8         # Increased slightly (Mac Air M1/M2 can usually handle 8 on ViT-Base)
):
    print("\n=== Starting Post-Pruning LoRA Recovery ===")

    # 1. Freeze Base Model
    for param in model.parameters():
        param.requires_grad = False

    # 2. Configure LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,       # Higher alpha helps LoRA learn faster
        target_modules=["query", "value", "dense"], 
        lora_dropout=0.05,
        bias="none",
        # REMOVED: modules_to_save=["classifier"] 
        # CRITICAL FIX: Do NOT retrain the head on small data!
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8,  # Effective Batch Size = 8 * 8 = 64
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=10,
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