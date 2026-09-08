import ast

from basic_call import HelloAgentsLLM


# 规划器提示词模板：让模型把复杂问题拆解成一个结构化的 Python 列表
PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
请特别注意：最后一个步骤必须是"根据前面的结果，汇总并输出最终答案"的步骤，这样整个计划才能收口到一个明确的结果上。

你的输出必须是一个 Python 列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划，```python 与 ``` 作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""


# 执行器提示词模板：让模型在当前上下文下，只专注解决"当前步骤"
EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""


class Planner:
    """规划器：负责把原始问题拆解成一个结构化的行动计划。"""

    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client

    def plan(self, question: str) -> list[str]:
        """
        根据用户问题生成一个行动计划。
        """
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        messages = [{"role": "user", "content": prompt}]

        print("--- 正在生成计划 ---")
        response_text = self.llm_client.think(messages=messages) or ""

        print(f"✅ 计划已生成:\n{response_text}")

        return self._parse_plan(response_text)

    def _parse_plan(self, response_text: str) -> list[str]:
        """
        从 LLM 输出中安全地解析出计划列表。
        依次尝试: ```python 代码块 -> ``` 代码块 -> 裸列表字符串。
        """
        plan_str = self._extract_list_str(response_text)
        if plan_str is None:
            print("❌ 未能从响应中提取到列表字符串。")
            return []

        try:
            plan = ast.literal_eval(plan_str)
            if isinstance(plan, list) and all(isinstance(s, str) for s in plan):
                return plan
            print("❌ 解析结果不是字符串列表。")
            return []
        except (ValueError, SyntaxError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []

    @staticmethod
    def _extract_list_str(response_text: str):
        """
        优先提取 ```python 与 ``` 之间的内容；若不存在，则退回 ``` 与 ``` 之间的内容；
        最后退回去除首尾空白后的整个响应文本。
        """
        # 1. 优先匹配带语言标记的代码块
        if "```python" in response_text:
            parts = response_text.split("```python", 1)
            if "```" in parts[1]:
                return parts[1].split("```", 1)[0].strip()

        # 2. 退回普通代码块
        if "```" in response_text:
            parts = response_text.split("```", 1)
            if "```" in parts[1]:
                return parts[1].split("```", 1)[0].strip()

        # 3. 退回整个响应文本
        return response_text.strip() or None


class Executor:
    """执行器：按计划逐步求解，并维护历史记录作为状态。"""

    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client

    def execute(self, question: str, plan: list[str]) -> str:
        """
        根据计划，逐步执行并解决问题，返回最终答案。
        """
        history = ""  # 用于存储历史步骤和结果的字符串
        final_answer = ""

        print("\n--- 正在执行计划 ---")

        for i, step in enumerate(plan):
            print(f"\n-> 正在执行步骤 {i+1}/{len(plan)}: {step}")

            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan=plan,
                history=history if history else "无",
                current_step=step,
            )

            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages) or ""

            # 更新历史记录，为下一步做准备
            history += f"步骤 {i+1}: {step}\n结果: {response_text}\n\n"

            # 记录本步结果；最后一步的结果即最终答案
            final_answer = response_text

            print(f"✅ 步骤 {i+1} 已完成，结果: {response_text}")

        return final_answer


class PlanAndSolveAgent:
    """协调者：组合 Planner 与 Executor，完成"先规划，后执行"。"""

    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client
        self.planner = Planner(self.llm_client)
        self.executor = Executor(self.llm_client)

    def run(self, question: str):
        """
        运行智能体的完整流程：先规划，后执行。
        """
        print(f"\n--- 开始处理问题 ---\n问题: {question}")

        # 1. 调用规划器生成计划
        plan = self.planner.plan(question)

        # 检查计划是否成功生成
        if not plan:
            print("\n--- 任务终止 ---\n无法生成有效的行动计划。")
            return None

        # 2. 调用执行器执行计划
        final_answer = self.executor.execute(question, plan)

        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")
        return final_answer


# --- 运行示例 ---
if __name__ == '__main__':
    llm_client = HelloAgentsLLM()
    agent = PlanAndSolveAgent(llm_client)

    answer = agent.run(
        "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。"
        "周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
    )

    print("\n=== 最终结果 ===")
    print(answer)
