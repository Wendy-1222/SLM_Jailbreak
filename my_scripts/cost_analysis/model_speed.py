import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# 配置：模型列表和测试文本
model_paths = [
    "path/to/model1",
    "path/to/model2",
    # 添加更多模型路径
]
test_prompt = "Hello, how are you today?"
max_output_tokens = 50  # 生成 token 数量
device = "cuda" if torch.cuda.is_available() else "cpu"

results = []

for model_path in model_paths:
    print(f"\nLoading model: {model_path}")
    
    # ----------------------------
    # 加载模型和 tokenizer
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    
    # ----------------------------
    # 测量 GPU memory
    # ----------------------------
    torch.cuda.reset_peak_memory_stats(device)
    
    # 先做一次 forward 准备缓存
    input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        model(input_ids)
    mem_bytes = torch.cuda.max_memory_allocated(device)
    
    # ----------------------------
    # 测量 prefill speed
    # ----------------------------
    start = time.time()
    with torch.no_grad():
        model(input_ids)
    prefill_time = time.time() - start
    prefill_tokens = input_ids.numel()
    prefill_speed = prefill_tokens / prefill_time
    
    # ----------------------------
    # 测量 decode speed
    # ----------------------------
    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_output_tokens,
            do_sample=False
        )
    decode_time = time.time() - start
    decode_tokens = output_ids.shape[1] - input_ids.shape[1]
    decode_speed = decode_tokens / decode_time
    
    # ----------------------------
    # 计算 end-to-end throughput
    # ----------------------------
    total_tokens = output_ids.shape[1]
    total_time = prefill_time + decode_time
    throughput = total_tokens / total_time
    
    # ----------------------------
    # 保存结果
    # ----------------------------
    results.append({
        "model": model_path,
        "mem_bytes": mem_bytes,
        "prefill_speed": prefill_speed,
        "decode_speed": decode_speed,
        "throughput": throughput
    })
    
    print(f"GPU Memory: {mem_bytes/1e6:.1f} MB")
    print(f"Prefill speed: {prefill_speed:.1f} tokens/s")
    print(f"Decode speed: {decode_speed:.1f} tokens/s")
    print(f"Throughput: {throughput:.1f} tokens/s")
    
# ----------------------------
# 最终输出所有模型的结果
# ----------------------------
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("model_efficiency.csv", index=False)
print("\nResults saved to model_efficiency.csv")
