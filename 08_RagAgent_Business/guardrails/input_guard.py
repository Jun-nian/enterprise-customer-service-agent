# guardrails/input_guard.py
# P0-C 输入侧护栏：提示词注入/越狱/敏感指令检测
# 在用户 Prompt 进入 LLM 推理前完成攻击意图检测（复用 vivo 7 类攻击范式）
# 命中策略：block（拒答）/ flag（降级为受限回答）
#
# 面试话术钩子：
#   "参照 vivo 输入侧安全护栏项目，在 Agent 入口挂轻量规则检测，
#    7 类攻击范式命中即拦截，外部输入一律按 HOSTILE 处理。"

import logging
from typing import Dict, Optional

from guardrails.patterns.attack_patterns import detect_attack

logger = logging.getLogger(__name__)

# 命中后的拒绝/降级响应模板
BLOCK_RESPONSE = ("抱歉，您的请求包含可疑指令，已被安全策略拦截。"
                  "如有疑问请联系企业支持。")
FLAG_RESPONSE = ("抱歉，您的请求可能包含风险内容，为保障企业信息安全，"
                 "我无法执行该请求。可尝试询问公司制度、FAQ 等合规问题。")


class InputGuardResult:
    """输入检测结果"""
    def __init__(self, passed: bool, attack_type: Optional[str] = None,
                 matched: Optional[str] = None, action: str = "allow",
                 response: Optional[str] = None):
        self.passed = passed
        self.attack_type = attack_type
        self.matched = matched
        self.action = action          # allow / block / flag
        self.response = response

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "attack_type": self.attack_type,
            "matched": self.matched,
            "action": self.action,
        }

    def __repr__(self):
        return f"<InputGuardResult {self.action} attack={self.attack_type} matched={self.matched!r}>"


def check_input(text: str, *, block_hard: bool = True, mode: str = "block") -> InputGuardResult:
    """检测用户输入是否命中攻击范式

    Args:
        text: 用户输入
        block_hard: 是否对高危类别（越狱/注入/窃取）直接拦截
        mode: block=命中即拒答 / flag=命中则降级为受限回答

    Returns:
        InputGuardResult
    """
    if not text or not text.strip():
        return InputGuardResult(passed=True, action="allow")

    attack_type, matched = detect_attack(text)
    if attack_type is None:
        return InputGuardResult(passed=True, action="allow")

    logger.warning(f"[input_guard] 命中攻击范式: {attack_type} (匹配: {matched!r})")

    # 高危类别：越狱 / 指令注入 / 数据窃取 —— 默认直接 block
    hard_categories = {"jailbreak", "prompt_injection", "exfiltration"}
    if mode == "block" or (block_hard and attack_type in hard_categories):
        return InputGuardResult(
            passed=False, attack_type=attack_type, matched=matched,
            action="block", response=BLOCK_RESPONSE,
        )
    # 降级：受限回答
    return InputGuardResult(
        passed=False, attack_type=attack_type, matched=matched,
        action="flag", response=FLAG_RESPONSE,
    )


def guard_input(text: str, config=None) -> InputGuardResult:
    """护栏入口（供工作流 pre-hook 调用）

    Args:
        text: 用户输入
        config: 可选 Config（读取护栏开关/模式）
    """
    from utils.config import Config as C
    if config is None:
        config = C
    if not getattr(config, "ENABLE_GUARDRAILS", False):
        return InputGuardResult(passed=True, action="allow")
    return check_input(text, mode=getattr(config, "GUARD_MODE", "block"))
