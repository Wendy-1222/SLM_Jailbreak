import requests
import re

def count_models_under_7b():
    # Hugging Face model hub API endpoint
    base_url = "https://huggingface.co/api/models"
    
    # Parameters for the API request
    params = {
        "filter": "text-generation-inference",  # 可选：筛选文本生成模型
        "limit": 10000  # 增加查询的模型数量限制
    }
    
    try:
        # 发送API请求
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # 检查请求是否成功
        models = response.json()
        
        # 统计小于等于7B参数的模型数量
        small_models = 0
        models_details = []
        
        for model in models:
            model_name = model['modelId']
            
            try:
                # 获取模型详细信息
                model_info_url = f"https://huggingface.co/api/models/{model_name}"
                model_info_response = requests.get(model_info_url)
                model_info_response.raise_for_status()
                model_info = model_info_response.json()
                
                # 提取参数信息
                model_card = model_info.get('cardData', {})
                model_card_text = str(model_card)
                
                # 使用正则表达式查找参数数量
                param_matches = re.findall(r'(\d+(?:\.\d+)?)\s*[bB]', model_card_text)
                
                if param_matches:
                    for match in param_matches:
                        param_size = float(match)
                        if param_size <= 7:
                            small_models += 1
                            models_details.append({
                                'name': model_name,
                                'params': param_size
                            })
                            break
            
            except Exception as model_error:
                print(f"Error processing model {model_name}: {model_error}")
        
        return small_models, models_details
    
    except Exception as e:
        print(f"Error fetching models: {e}")
        return 0, []

# 运行统计
total_small_models, small_model_details = count_models_under_7b()

print(f"总计小于等于7B参数的模型数量: {total_small_models}")
print("\n小于等于7B的模型详情:")
for model in small_model_details:
    print(f"模型名称: {model['name']}, 参数量: {model['params']}B")