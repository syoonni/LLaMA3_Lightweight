import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch.nn.functional as F
import math
import time
from tqdm import tqdm

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)

def load_peft_model(base_model_path, peft_path):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, peft_path)
    return model

def evaluate_peft_model(base_model_path, peft_path, test_text):
    start_time = time.time()
    
    # 모델 로드 시간 측정
    model_load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = load_peft_model(base_model_path, peft_path)
    model_load_time = time.time() - model_load_start
    
    model.eval()
    
    # 테스트 텍스트를 문장들로 분리
    sentences = [s for s in test_text.split('\n') if s.strip()]
    total_loss = 0
    total_tokens = 0
    
    # 추론 시간 측정
    inference_start = time.time()
    
    for sentence in tqdm(sentences, desc="Evaluating PEFT model"):
        if not sentence or sentence.isspace():
            continue
            
        # 전체 토큰화
        tokens = tokenizer.encode(sentence)
        if len(tokens) < 2:  # 최소 2개의 토큰 필요
            continue
            
        # 입력은 전체 시퀀스, labels도 전체 시퀀스로 설정
        input_ids = torch.tensor(tokens).unsqueeze(0).to(model.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            
        # 실제 예측한 토큰 수 (첫 토큰을 제외한 나머지)
        pred_tokens = input_ids.size(1) - 1  # 첫 토큰은 예측에서 제외
        total_loss += loss.item() * pred_tokens
        total_tokens += pred_tokens
    
    inference_time = time.time() - inference_start
    total_time = time.time() - start_time
    
    # 평균 loss 계산 (실제 예측한 토큰에 대해서만)
    eval_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    
    # perplexity 계산
    try:
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float('inf')
    
    # metrics 반환
    metrics = {
        "eval_loss": eval_loss,
        "eval_tokens": total_tokens,  # 실제 예측한 토큰 수
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
    
    # 테스트 텍스트 예시 (실제 평가할 텍스트로 교체 필요)
    test_text = """Once upon a time, in a warm and sunny place, there was a big pit..."""
    
    print("Starting PEFT model evaluation...")
    metrics = evaluate_peft_model(base_model_path, peft_path, test_text)
    
    print("\nPEFT Model Evaluation Results:")
    print(f"Loss: {metrics['eval_loss']:.6f}")
    print(f"Perplexity: {metrics['perplexity']:.6f}")
    print(f"Total evaluated tokens: {metrics['eval_tokens']}")
    print("\nTiming Information:")
    print(f"Model Load Time: {metrics['model_load_time']:.2f} seconds")
    print(f"Inference Time: {metrics['inference_time']:.2f} seconds")
    print(f"Total Time: {metrics['total_time']:.2f} seconds")
    print(f"Processing Speed: {metrics['tokens_per_second']:.2f} tokens/second")

if __name__ == "__main__":
    main()