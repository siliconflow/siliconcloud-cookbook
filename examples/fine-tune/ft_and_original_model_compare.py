from openai import OpenAI
client = OpenAI(
    api_key="your_api_key", # 从https://cloud.siliconflow.cn/account/ak获取
    base_url="https://api.siliconflow.cn/v1"
)
 
def compare_model(word):
    qwen2_5_7B_original_messages = [
        {"role": "system", "content": f"""# 角色
            你是一位新潮评论家，你年轻、批判，又深刻；
            你言辞犀利而幽默，擅长一针见血得表达隐喻，对现实的批判讽刺又不失文雅；
            你的行文风格和"Oscar Wilde" "鲁迅" "林语堂"等大师高度一致

            # 任务
            ## 金句诠释
            用特殊视角来全新得诠释给定的汉语词汇；
            敏锐得抓住给定的词汇的本质，用“辛辣的讽刺”“一针见血的评论”的风格构造包含隐喻又直达本质的「金句」
            例如：“委婉”： "刺向他人时, 决定在剑刃上撒上止痛药。"
            
            ### 结果示例：
            {{委婉：刺向他人时, 决定在剑刃上撒上止痛药。}}
        """},
        {"role": "user", "content": f"{word}"},
    ]

    qwen2_5_7B_fine_tuned_messages = [
        {"role": "system", "content": "你是智说新语生成器。"},
        {"role": "user", "content": f"{word}"},
    ]

    # 使用原始的Qwen2.5-7B-Instruct模型
    qwen2_5_7B_original_response = client.chat.completions.create(
        # 模型名称，从 https://cloud.siliconflow.cn/models 获取
        model="Qwen/Qwen2.5-7B-Instruct", 
        messages=qwen2_5_7B_original_messages,
        stream=True,
        max_tokens=4096
    )

    print('\033[31m使用基于Qwen2.5-7B-Instruct的原始模型:\033[0m')
    for chunk in qwen2_5_7B_original_response: 
        print(chunk.choices[0].delta.content, end='')

    # 使用基于Qwen2.5-7B-Instruct+智说新语语料微调后的模型
    qwen2_5_7B_fine_tuned_response = client.chat.completions.create(
        # 模型名称，从 https://cloud.siliconflow.cn/fine-tune 获取对应的微调任务
        model="ft:LoRA/Qwen/Qwen2.5-7B-Instruct:{your-complete-fine-tune-model-name}", 
        messages=qwen2_5_7B_fine_tuned_messages,
        stream=True,
        max_tokens=4096
    )

    print('\n\033[32m使用基于Qwen2.5-7B-Instruct+智说新语语料微调后的模型:\033[0m')
    print(f"{word}：", end='')
    for chunk in qwen2_5_7B_fine_tuned_response:
        print(chunk.choices[0].delta.content, end='')
        
if __name__ == '__main__':
    words = ['降维打击', '新时代', '创新', '协作']
    for word in words:
        compare_model(word)
        print('\n')
