# guardrails/output_guard.py
# P0-C 输出侧护栏：PII 脱敏
# 对 LLM 输出进行敏感信息检测与脱敏：手机号、邮箱、身份证号、银行卡号等
# 保障"审计日志不记全量敏感数据"，输出到用户前完成脱敏。
#
# 面试话术钩子：
#   "输出侧 PII 脱敏——员工信息查询结果中的电话/邮箱等敏感字段，
#    在落库审计或跨系统输出前统一脱敏，避免敏感数据外泄。"

import logging
import re

logger = logging.getLogger(__name__)

# 脱敏规则（名称 → (正则, 脱敏函数)）
# 兼容 11 位手机号与模拟数据中的 10 位短号码（1[3-9] 开头）
_PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{8,9})(?!\d)")
_IDCARD_RE = re.compile(r"(?<!\d)(\d{17}[\dXx])(?!\d)")
_EMAIL_RE = re.compile(r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
_BANKCARD_RE = re.compile(r"(?<!\d)(\d{16,19})(?!\d)")


def _mask_phone(m: re.Match) -> str:
    digits = m.group(1)
    if len(digits) == 11:
        return f"{digits[:3]}****{digits[-4:]}"
    return f"{digits[:3]}****{digits[-3:]}"


def _mask_idcard(m: re.Match) -> str:
    return f"{m.group(1)[:4]}**********{m.group(1)[-4:]}"


def _mask_email(m: re.Match) -> str:
    local, domain = m.group(1), m.group(2)
    masked_local = local[0] + "***" if len(local) > 1 else "***"
    return f"{masked_local}@{domain}"


def _mask_bankcard(m: re.Match) -> str:
    return f"{m.group(1)[:4]}****{m.group(1)[-4:]}"


# 规则顺序：先长后短，避免部分匹配
_MASK_RULES = [
    ("phone", _PHONE_RE, _mask_phone),
    ("idcard", _IDCARD_RE, _mask_idcard),
    ("bankcard", _BANKCARD_RE, _mask_bankcard),
    ("email", _EMAIL_RE, _mask_email),
]


def mask_pii(text: str, *, mask_email: bool = True) -> str:
    """对文本中的 PII 进行脱敏

    Args:
        text: 原始文本
        mask_email: 是否同时脱敏邮箱（默认 True）

    Returns:
        str: 脱敏后的文本
    """
    if not text:
        return text
    result = text
    for name, pattern, repl in _MASK_RULES:
        if name == "email" and not mask_email:
            continue
        result = pattern.sub(repl, result)
    return result


def guard_output(text: str, config=None) -> str:
    """输出护栏入口（供工作流 post-hook 调用）

    Args:
        text: LLM 生成的回答
        config: 可选 Config（读取护栏开关）

    Returns:
        str: 脱敏后的回答
    """
    from utils.config import Config as C
    if config is None:
        config = C
    if not getattr(config, "ENABLE_GUARDRAILS", False):
        return text
    return mask_pii(text)


# 供测试/调试
if __name__ == "__main__":
    samples = [
        "张建国电话是1380001001，邮箱 zhangjianguo@xinrui.com",
        "身份证号 110101199003071234",
        "银行卡号 6222021234567890123",
        "正常内容，不含敏感信息",
    ]
    for s in samples:
        print(f"  原始: {s}")
        print(f"  脱敏: {mask_pii(s)}")
