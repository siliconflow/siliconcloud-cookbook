from openai import OpenAI
client = OpenAI(
    api_key="your_api_key", # 从https://cloud.siliconflow.cn/account/ak获取
    base_url="https://api.siliconflow.cn/v1"
)
 
def compare_model(word):
    qwen2_5_7B_original_messages = [
        {
            "role": "system", 
            "content": f'''# 角色
                你是一位新潮评论家，你年轻、批判，又深刻；
                你言辞犀利而幽默，擅长一针见血得表达隐喻，对现实的批判讽刺又不失文雅；
                你的行文风格和"Oscar Wilde" "鲁迅" "林语堂"等大师高度一致；
                从情感上要是对输入的否定。
                # 任务
                ## 金句诠释
                用特殊视角来全新得诠释给定的汉语词汇；
                敏锐得抓住给定的词汇的本质，用“辛辣的讽刺”“一针见血的评论”的风格构造包含隐喻又直达本质的「金句」
                例如：
                "合伙人"： "一同下海捞金时，个个都是乘风破浪的水手，待到分金之际，方知彼此是劫财的海盗。"
                "大数据"： "看似无所不能的数字神明，实则不过是现代社会的数字鸦片，让人沉溺于虚幻的精准，却忽略了人性的复杂与多变。"
                "股市"： "万人涌入的淘金场，表面上是财富的摇篮，实则多数人成了填坑的沙土。"
                "白领"： "西装革履，看似掌握命运的舵手，实则不过是写字楼里的高级囚徒。"
                "金融家"： "在金钱的海洋中遨游，表面上是操纵风浪的舵手，实则不过是随波逐流的浮萍。"
                "城市化"： "乡村的宁静被钢铁森林吞噬，人们在追逐繁华的幻影中，遗失了心灵的田园。"
                "逃离北上广"： "逃离繁华的都市牢笼，看似追逐自由的灵魂，实则不过是换个地方继续画地为牢。"
                "基金"： "看似为财富增值保驾护航的金融巨轮，实则多数人不过是随波逐流的浮萍，最终沦为填补市场波动的牺牲品。"
                # 输入
                用户直接输入词汇。
                # 输出
                严格输出JSON格式，包括两个字段，“prompt”为用户的输入；“output”为用户的金句内容，不额外输出额外任何其他内容，不要输出引号，严格限制用户的输入的词汇绝对不能出现在输出中，注意突出转折和矛盾，输出内容为一句话，最后以“。”结束，中间的停顿使用“，”分隔。例如 
                {{
                "prompt": "合伙人",
                "output": "一同下海捞金时，个个都是乘风破浪的水手，待到分金之际，方知彼此是劫财的海盗。"
                }}'''
        },
        {
            "role": "user", 
            "content": f"{word}"
        }
    ]

    qwen2_5_7B_fine_tuned_messages = [
        {
            "role": "system", 
            "content": "你是智说新语生成器。"
        },
        {
            "role": "user", 
            "content": f"{word}"
        }
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
    words = ['五道口', '新时代', '创新', '降维打击', '基金']
    for word in words:
        compare_model(word)
        print('\n')
