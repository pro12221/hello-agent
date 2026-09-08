from basic_call import HelloAgentsLLM


# 生成器提示词：让模型给出一个初步答案
GENERATOR_PROMPT_TEMPLATE = """
你是一位经验丰富的AI助手。请认真回答下面的问题，并给出你的答案与简要推理过程。

问题: {question}
"""


# 反思器提示词：让模型批判性地审视自己刚才的答案
REFLECTOR_PROMPT_TEMPLATE = """
你是一位极其严格的评审专家。请批判性地审查下面给出的答案，找出其中的错误、遗漏、逻辑漏洞或表述不清之处。

# 原始问题:
{question}

# 待审查的答案:
{answer}

请从以下几个方面进行审查：
1. 答案是否正确、完整？
2. 推理过程是否有逻辑漏洞或计算错误？
3. 是否存在遗漏的关键信息或假设？

如果答案已经很好、没有需要改进的地方，请只输出一行：无需修改
否则，请具体指出问题所在，并说明应该如何改进。
"""


# 精炼器提示词：让模型根据反思结果，修正并输出一个更好的答案
REFINER_PROMPT_TEMPLATE = """
你是一位严谨的AI助手。你之前给出了一个答案，但评审专家指出了其中的不足。请你根据评审意见，修正错误、补全遗漏，重新给出一个更准确、更完整的答案。

# 原始问题:
{question}

# 你之前的答案:
{answer}

# 评审专家的意见:
{critique}

请仅输出修正后的最终答案与推理过程。
"""


class ReflectionAgent:
    """反思智能体：先给出答案，再自我批判，最后根据批判修正答案，循环往复。"""

    def __init__(self, llm_client: HelloAgentsLLM, max_iterations: int = 3):
        self.llm_client = llm_client
        self.max_iterations = max_iterations

    def run(self, question: str):
        """
        运行完整的反思流程：生成 -> 反思 -> 修正 -> (循环) -> 最终答案。
        """
        print(f"\n--- 开始处理问题 ---\n问题: {question}")

        # 1. 生成初步答案
        answer = self._generate(question)
        print(f"\n✅ 初步答案已生成:\n{answer}")

        # 2. 进入 反思 -> 修正 循环
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1} 轮反思 ---")

            # 2.1 反思：批判当前答案
            critique = self._reflect(question, answer)
            print(f"✅ 评审意见:\n{critique}")

            # 2.2 若评审认为无需修改，则提前终止
            if self._is_satisfied(critique):
                print("\n🎉 评审认为答案已无问题，流程提前结束。")
                break

            # 2.3 修正：根据意见生成更好的答案
            answer = self._refine(question, answer, critique)
            print(f"✅ 修正后的答案:\n{answer}")

        print(f"\n--- 任务完成 ---\n最终答案:\n{answer}")
        return answer

    def _generate(self, question: str) -> str:
        prompt = GENERATOR_PROMPT_TEMPLATE.format(question=question)
        messages = [{"role": "user", "content": prompt}]
        return self.llm_client.think(messages=messages) or ""

    def _reflect(self, question: str, answer: str) -> str:
        prompt = REFLECTOR_PROMPT_TEMPLATE.format(question=question, answer=answer)
        messages = [{"role": "user", "content": prompt}]
        return self.llm_client.think(messages=messages) or ""

    def _refine(self, question: str, answer: str, critique: str) -> str:
        prompt = REFINER_PROMPT_TEMPLATE.format(
            question=question, answer=answer, critique=critique
        )
        messages = [{"role": "user", "content": prompt}]
        return self.llm_client.think(messages=messages) or ""

    @staticmethod
    def _is_satisfied(critique: str) -> bool:
        """
        根据评审意见判断答案是否已经令人满意。
        当反思器输出"无需修改"之类的信号时，认为可以提前停止。
        """
        stop_keywords = ["无需修改", "没有问题", "无需改进", "已经很", "没有错误"]
        return any(kw in critique for kw in stop_keywords)


# --- 运行示例 ---
if __name__ == '__main__':
    llm_client = HelloAgentsLLM()
    agent = ReflectionAgent(llm_client)

    answer = agent.run(
        "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。"
        "周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
    )

    print("\n=== 最终结果 ===")
    print(answer)
