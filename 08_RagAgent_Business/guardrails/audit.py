# guardrails/audit.py
# P0-C 脱敏审计日志
# 记录高危工具调用（含越权尝试、攻击命中），但**不记录全量敏感数据**：
#   - 工具调用：记录工具名 + 参数摘要（截断 + PII 脱敏）
#   - 攻击命中：记录攻击类别 + 命中模式（不记完整 prompt）
#   - 越权尝试：记录用户角色 + 工具 + 原因
# 审计日志独立于应用日志，写入 output/audit.log（轮转）。

import logging
import json
import os
import time
import uuid

logger = logging.getLogger(__name__)

_AUDIT_FILE = "output/audit.log"


def _get_audit_logger() -> logging.Logger:
    """获取审计专用 logger（独立文件、独立级别）"""
    name = "agent_audit"
    audit_logger = logging.getLogger(name)
    if not audit_logger.handlers:
        audit_logger.setLevel(logging.INFO)
        audit_logger.propagate = False
        try:
            from concurrent_log_handler import ConcurrentRotatingFileHandler
            os.makedirs(os.path.dirname(_AUDIT_FILE), exist_ok=True)
            handler = ConcurrentRotatingFileHandler(
                _AUDIT_FILE, maxBytes=5 * 1024 * 1024, backupCount=2,
                encoding="utf-8",
            )
        except Exception:
            handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        audit_logger.addHandler(handler)
    return audit_logger


def _sanitize_args(args: dict) -> dict:
    """对工具参数做脱敏摘要（截断 + PII 脱敏，不落全量敏感数据）"""
    import guardrails.output_guard as og
    clean = {}
    for k, v in (args or {}).items():
        if isinstance(v, str):
            # 截断 + 脱敏
            s = v[:200]
            clean[k] = og.mask_pii(s)
        else:
            clean[k] = str(v)[:200]
    return clean


def audit_event(event_type: str, *, user_id: str = "", role: str = "",
                tool_name: str = "", args: dict = None, reason: str = "",
                attack_type: str = "", matched: str = "", action: str = "") -> None:
    """写入审计日志

    Args:
        event_type: tool_call / attack_block / authz_denied
        user_id: 用户标识（不落全量，仅记录）
        role: 解析出的角色
        tool_name: 工具名
        args: 工具参数（自动脱敏+截断）
        reason: 拒绝原因
        attack_type: 攻击类别
        matched: 命中的攻击模式（仅模式文本）
        action: block / flag / allow
    """
    try:
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": event_type,
            "audit_id": uuid.uuid4().hex[:12],
            "user": user_id[:64] if user_id else "",
            "role": role,
        }
        if tool_name:
            record["tool"] = tool_name
        if args:
            record["args"] = _sanitize_args(args)
        if reason:
            record["reason"] = reason[:300]
        if attack_type:
            record["attack"] = attack_type
        if matched:
            record["matched"] = matched[:120]     # 仅记录命中模式，不记完整 prompt
        if action:
            record["action"] = action

        _get_audit_logger().info(json.dumps(record, ensure_ascii=False))
    except Exception as e:
        logger.error(f"[audit] 审计写入失败: {e}")


def audit_tool_call(tool_name: str, args: dict, user_id: str, role: str, allowed: bool, reason: str = "") -> None:
    """工具调用审计（含越权事件）"""
    event_type = "authz_denied" if not allowed else "tool_call"
    audit_event(
        event_type, user_id=user_id, role=role, tool_name=tool_name,
        args=args, reason=reason,
    )


def audit_attack(prompt_sample: str, attack_type: str, matched: str, action: str, user_id: str = "") -> None:
    """攻击命中审计（不记录完整 prompt，仅类型 + 命中模式）"""
    audit_event(
        "attack_block", user_id=user_id, attack_type=attack_type,
        matched=matched, action=action,
    )
