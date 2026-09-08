from openai import OpenAI # 兼容openAI的客户端
from typing import List, Dict

# 默认配置（硬编码，参考 OpenClaude 配置文件 ~/.openclaude/.openclaude-profile.json）
DEFAULT_MODEL = "glm-5.3"
DEFAULT_API_KEY = "xxx"
DEFAULT_BASE_URL = "https://api.modelverse.cn/v1"
DEFAULT_TIMEOUT = 60

# 简单封装类，作用是统一调用不同供应商的大模型接口，并默认走流式输出。
class HelloAgentsLLM: 
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则使用代码中硬编码的默认配置。
        """
        self.model = model or DEFAULT_MODEL # 如果 model 是 None 或空串，回退到硬编码默认值
        apiKey = apiKey or DEFAULT_API_KEY
        baseUrl = baseUrl or DEFAULT_BASE_URL
        timeout = timeout or DEFAULT_TIMEOUT

        if not all([self.model, apiKey, baseUrl]): # 检查三项是否都非空
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在代码默认配置中定义。")

        # 创建实际的客户端
        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True, # 启用流式输出
            )
            
            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                if not chunk.choices:# 某些chunk可能没有choices字段，直接跳过
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None

# --- 客户端使用示例 ---
if __name__ == '__main__':
    try:
        llmClient = HelloAgentsLLM()
        
        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]
        
        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)

