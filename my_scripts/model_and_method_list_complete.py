method_list = ['DirectRequest', 'HumanJailbreaks', 'PAP', 'PAIR', 'GCG', 'AutoPrompt', 'PEZ', 'GBDA', 'UAT']

llama_32_series = "llama3_2_1b_instruct,llama3_2_3b_instruct"
deepseek_r1_series = "DeepSeek-R1-Distill-Qwen-1.5B,DeepSeek-R1-Distill-Qwen-7B,DeepSeek-R1-Distill-Llama-8B"
qwen_series = "qwen_1_8b_chat,qwen_7b_chat,qwen1_5_0_5b_chat,qwen1_5_1_8b_chat,qwen1_5_4b_chat,qwen1_5_7b_chat,qwen2_0_5b_instruct,qwen2_1_5b_instruct,qwen2_7b_instruct,qwen2_5_0_5b_instruct,qwen2_5_1_5b_instruct,qwen2_5_3b_instruct,qwen2_5_7b_instruct"
pythia_series = "pythia_14m,pythia_31m,pythia_70m,pythia_160m,pythia_410m,pythia_1b,pythia_1_4b,pythia_2_8b,pythia_6_9b"
phi_series = "phi_1_5,phi_2,phi_3_mini_4k_instruct,phi_3_mini_128k_instruct,phi_3_small_8k_instruct,phi_3_small_128k_instruct,phi_3_5_mini_instruct"
stablelm_series = "stablelm-2-zephyr-1_6b,stablelm-2-1_6b-chat,stablelm-zephyr-3b"
tiny_llama_series = "tinyllama-1.1B-chat-v0.1,tinyllama-1.1B-chat-v0.2,tinyllama-1.1B-chat-v0.3,tinyllama-1.1B-chat-v0.4,tinyllama-1.1B-chat-v0.5,tinyllama-1.1B-chat-v0.6,tinyllama-1.1B-chat-v1.0"
mobile_llama_series="mobilellama-1.4B-chat,mobilellama-2.7B-chat"
mobi_llama_series="mobillama-0.5B-chat,mobillama-1B-chat"
gemma_series="gemma-2b-it,gemma-7b-it,gemma-1.1-2b-it,gemma-1.1-7b-it,gemma-2-2b-it,recurrentgemma-2b-it"
minicpm_series="minicpm-1B-sft-bf16,minicpm-S-1B-sft,minicpm-2B-sft-bf16,minicpm-2B-sft-fp32,minicpm-2B-sft-int4,minicpm-2B-dpo-bf16,minicpm-2B-dpo-fp16,minicpm-2B-dpo-fp32,minicpm-2B-dpo-int4,minicpm-2B-128k,minicpm3-4B"
h2o_danube_series="h2o-danube-1.8b-sft,h2o-danube-1.8b-chat,h2o-danube2-1.8b-sft,h2o-danube2-1.8b-chat,h2o-danube3-500m-chat,h2o-danube3-4b-chat"
fox_series = "fox-1-1.6B-Instruct-v0.1"
smollm_series = "smollm-135M-instruct,smollm-360M-instruct,smollm-1.7B-instruct,smollm2-135M-instruct,smollm2-360M-instruct,smollm2-1.7B-instruct"
dclm_series = "DCLM-1B-IT"
dolly_series = "dolly-v1-6b,dolly-v2-3b,dolly-v2-7b"
olmo_series = "OLMo-7B-SFT-hf,OLMo-7B-Instruct-hf"
qwen2_5_quantized_series = "qwen2_5_0_5b_instruct_gptq_int4,qwen2_5_1_5b_instruct_gptq_int4,qwen2_5_3b_instruct_gptq_int4,qwen2_5_7b_instruct_gptq_int4,qwen2_5_0_5b_instruct_gptq_int8,qwen2_5_1_5b_instruct_gptq_int8,qwen2_5_3b_instruct_gptq_int8,qwen2_5_7b_instruct_gptq_int8,qwen2_5_0_5b_instruct_awq,qwen2_5_1_5b_instruct_awq,qwen2_5_3b_instruct_awq,qwen2_5_7b_instruct_awq"
qwen3_series = "qwen3_0_6b,qwen3_1_7b,qwen3_4b"

# 转换成列表
llama_32_series = llama_32_series.split(',')
deepseek_r1_series = deepseek_r1_series.split(',')
qwen_series = qwen_series.split(',')
# pythia_series = pythia_series.split(',')
phi_series = phi_series.split(',')
stablelm_series = stablelm_series.split(',')
tiny_llama_series = tiny_llama_series.split(',')
mobile_llama_series = mobile_llama_series.split(',')
mobi_llama_series = mobi_llama_series.split(',')
gemma_series = gemma_series.split(',')
minicpm_series = minicpm_series.split(',')
h2o_danube_series = h2o_danube_series.split(',')
fox_series = fox_series.split(',')
smollm_series = smollm_series.split(',')
dclm_series = dclm_series.split(',')
dolly_series = dolly_series.split(',')
olmo_series = olmo_series.split(',')
qwen2_5_quantized_series = qwen2_5_quantized_series.split(',')
qwen3_series = qwen3_series.split(',')

model_list = ['llama2_7b', 'vicuna_7b_v1_5']
model_list += llama_32_series + deepseek_r1_series + qwen_series + pythia_series + phi_series + stablelm_series + tiny_llama_series + mobile_llama_series + mobi_llama_series + gemma_series + minicpm_series + h2o_danube_series + fox_series + smollm_series + dclm_series + dolly_series + olmo_series + qwen2_5_quantized_series + qwen3_series
