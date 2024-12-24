import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch.nn.functional as F
import math
import time
from tqdm import tqdm

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)

def calculate_perplexity(model, tokenizer, sentence):
    """Trainer와 유사한 방식으로 perplexity 계산"""
    tokens = tokenizer.encode(sentence)
    if len(tokens) < 2:
        return 0, 0
        
    # 전체 시퀀스를 입력으로 사용
    input_ids = torch.tensor(tokens).unsqueeze(0).to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, :-1, :]  # 마지막 위치 제외
        labels = input_ids[:, 1:]  # 두 번째 토큰부터
        
        # Loss 계산
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=tokenizer.pad_token_id
        )
    
    return loss.item(), len(tokens) - 1

def evaluate_test_set(model_path, test_text):
    start_time = time.time()
    
    model_load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model_load_time = time.time() - model_load_start
    
    model.eval()
    
    sentences = [s for s in test_text.split('\n') if s.strip()]
    total_loss = 0
    total_tokens = 0
    
    inference_start = time.time()
    
    for sentence in tqdm(sentences, desc="Evaluating base model"):
        if not sentence or sentence.isspace():
            continue
            
        loss, num_tokens = calculate_perplexity(model, tokenizer, sentence)
        total_loss += loss * num_tokens
        total_tokens += num_tokens
    
    inference_time = time.time() - inference_start
    total_time = time.time() - start_time
    
    eval_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    
    try:
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float('inf')
    
    metrics = {
        "eval_loss": eval_loss,
        "eval_tokens": total_tokens,
        "perplexity": perplexity,
        "model_load_time": model_load_time,
        "inference_time": inference_time,
        "total_time": total_time,
        "tokens_per_second": total_tokens / inference_time if inference_time > 0 else 0
    }
    
    return metrics

def evaluate_peft_model(base_model_path, peft_path, test_text):
    start_time = time.time()
    
    model_load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, peft_path)
    model_load_time = time.time() - model_load_start
    
    model.eval()
    
    sentences = [s for s in test_text.split('\n') if s.strip()]
    total_loss = 0
    total_tokens = 0
    
    inference_start = time.time()
    
    for sentence in tqdm(sentences, desc="Evaluating PEFT model"):
        if not sentence or sentence.isspace():
            continue
            
        loss, num_tokens = calculate_perplexity(model, tokenizer, sentence)
        total_loss += loss * num_tokens
        total_tokens += num_tokens
    
    inference_time = time.time() - inference_start
    total_time = time.time() - start_time
    
    eval_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    
    try:
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float('inf')
    
    metrics = {
        "eval_loss": eval_loss,
        "eval_tokens": total_tokens,
        "perplexity": perplexity,
        "model_load_time": model_load_time,
        "inference_time": inference_time,
        "total_time": total_time,
        "tokens_per_second": total_tokens / inference_time if inference_time > 0 else 0
    }
    
    return metrics

def main():
    base_model_path = "/home/ehost/syoon/EdgeAI/Llama-3.2-1B"
    peft_path = "/home/ehost/syoon/EdgeAI/llama_checkpoints/final_model"
    
    test_text = """Once upon a time, in a warm and sunny place, there was a big pit..."""
    
    print("\nStarting base model evaluation...")
    base_metrics = evaluate_test_set(base_model_path, test_text)
    
    print("\nBase Model Evaluation Results:")
    print(f"Loss: {base_metrics['eval_loss']:.6f}")
    print(f"Perplexity: {base_metrics['perplexity']:.6f}")
    print(f"Total evaluated tokens: {base_metrics['eval_tokens']}")
    print("\nTiming Information:")
    print(f"Model Load Time: {base_metrics['model_load_time']:.2f} seconds")
    print(f"Inference Time: {base_metrics['inference_time']:.2f} seconds")
    print(f"Total Time: {base_metrics['total_time']:.2f} seconds")
    print(f"Processing Speed: {base_metrics['tokens_per_second']:.2f} tokens/second")
    
    print("\nStarting PEFT model evaluation...")
    peft_metrics = evaluate_peft_model(base_model_path, peft_path, test_text)
    
    print("\nPEFT Model Evaluation Results:")
    print(f"Loss: {peft_metrics['eval_loss']:.6f}")
    print(f"Perplexity: {peft_metrics['perplexity']:.6f}")
    print(f"Total evaluated tokens: {peft_metrics['eval_tokens']}")
    print("\nTiming Information:")
    print(f"Model Load Time: {peft_metrics['model_load_time']:.2f} seconds")
    print(f"Inference Time: {peft_metrics['inference_time']:.2f} seconds")
    print(f"Total Time: {peft_metrics['total_time']:.2f} seconds")
    print(f"Processing Speed: {peft_metrics['tokens_per_second']:.2f} tokens/second")

if __name__ == "__main__":
    main()