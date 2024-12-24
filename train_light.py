import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from data_pipeline import TinyStoriesDataset, TinyStoriesDatasetFactory
from torch.utils.data import Subset
import numpy as np
import os
import math

np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    mask = labels != -100
    logits = logits[mask]
    labels = labels[mask]
    loss = F.cross_entropy(
        torch.tensor(logits).view(-1, logits.shape[-1]),
        torch.tensor(labels).view(-1)
    )
    perplexity = math.exp(loss.item())
    return {"perplexity": perplexity}

def create_limited_dataset(file_path, tokenizer, max_length, limit):
    full_dataset = TinyStoriesDataset(
        file_path=file_path,
        tokenizer=tokenizer,
        max_length=max_length
    )
    indices = range(min(len(full_dataset), limit))
    return Subset(full_dataset, indices)

def train_llama3():
    batch_size = 1
    learning_rate = 1e-6
    num_epochs = 3
    model_path = "Llama-3.2-1B"
    max_length = 16  # 시퀀스 길이 감소

    # 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )

    # 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        padding_side="right",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 제한된 크기의 데이터셋 생성
    train_dataset = create_limited_dataset(
        "Dataset/TinyStoriesV2-GPT4-test.txt",
        tokenizer,
        max_length,
        100000
    )
    
    test_dataset = create_limited_dataset(
        "Dataset/TinyStoriesV2-GPT4-test.txt",
        tokenizer,
        max_length,
        1000
    )

    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )

    model = prepare_model_for_kbit_training(model)

    # LoRA 설정
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.3,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    checkpoint_dir = "llama_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        weight_decay=0.01,
        fp16=True,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        warmup_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="perplexity",
        greater_is_better=False,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        max_grad_norm=0.3,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    print("\nSample generation before training:")
    prompt = "Once upon a time, there was a little"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    print("\nMetrics before training:")
    metrics = trainer.evaluate()
    print(metrics)

    print("\nTraining...")
    trainer_result = trainer.train()

    print("\nSaving model...")
    trainer.save_model(os.path.join(checkpoint_dir, "final_model"))

    print("\nSample generation after training:")
    outputs = model.generate(**inputs, max_length=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    print("\nEvaluating on test set...")
    metrics = trainer.evaluate()
    print(f"Test set metrics: {metrics}")

if __name__ == "__main__":
    train_llama3()