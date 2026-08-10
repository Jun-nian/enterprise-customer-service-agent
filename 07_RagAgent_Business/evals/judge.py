# evals/judge.py
# LLM-as-judge 回答质量判定：判断回答是否覆盖金标准事实
# 复用 demoRagAgent.create_chain 的 prompt 缓存与结构化输出机制

import sys
import os

# 将项目根目录加入 sys.path，以便导入 utils 和 mcp_server
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pydantic import BaseModel, Field
from demoRagAgent import create_chain
from utils.config import Config


class AnswerJudgeScore(BaseModel):
    """答案评审结构化输出"""
    binary_score: str = Field(description="Answer coverage score 'yes' or 'no'")


def judge_answer(llm_chat, question: str, answer: str, golden_contains: list) -> bool:
    """判断回答是否覆盖全部金标准事实

    Args:
        llm_chat: 聊天模型实例
        question: 用户问题
        answer: 智能客服的最终回答
        golden_contains: 金标准事实子串列表

    Returns:
        bool: True=覆盖全部事实，False=至少缺失一条
    """
    if not answer or not golden_contains:
        return False

    # 把金标准子串组织成编号列表喂给评审员
    golden_facts = "\n".join(f"{i+1}. {g}" for i, g in enumerate(golden_contains))

    judge_chain = create_chain(llm_chat, Config.PROMPT_TEMPLATE_TXT_JUDGE, AnswerJudgeScore)
    try:
        result = judge_chain.invoke({"question": question, "answer": answer, "golden_facts": golden_facts})
        score = result.binary_score.lower()
        return score == "yes"
    except Exception as e:
        print(f"[judge] 评审失败: {e}")
        return False
