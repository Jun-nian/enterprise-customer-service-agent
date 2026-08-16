# guardrails/authz.py
# P0-C 工具越权校验（Authorization）
# 对敏感工具（search_employees 等含 PII 的工具）做身份 + 归属校验：
#   - 未认证用户（anonymous）不允许访问敏感工具
#   - 普通用户仅能访问与自身相关的数据（数据归属校验）
#   - HR/管理员角色可跨部门访问
#
# 设计（面试话术钩子）：
#   "在 mcp_server/tools_impl.py 单一事实来源统一挂 AuthZ 校验，
#    内部 LangGraph Agent 与外部 MCP 客户端共享同一套鉴权规则。"

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 角色权限表：工具 → 允许访问的角色集合
TOOL_ROLE_PERMISSIONS = {
    "retrieve":        {"anonymous", "user", "hr", "admin"},
    "search_faq":      {"anonymous", "user", "hr", "admin"},
    "search_orders":   {"user", "hr", "admin"},
    "search_tickets":  {"user", "hr", "admin"},
    "search_employees": {"hr", "admin"},      # 员工信息含 PII，仅 HR/管理员
}

# 角色权重（用于归属校验的级别判断）
_ROLE_LEVEL = {"anonymous": 0, "user": 1, "hr": 2, "admin": 3}

# 每个工具对应数据中的"归属字段"（用于 owner 校验）
# 实际场景应从会话上下文获取当前用户信息；这里用 user_id 做简化演示。
_TOOL_OWNER_FIELD = {
    "search_tickets": "applicant",    # 工单申请人 = 归属人
    "search_orders": "customer",      # 订单客户 = 归属人
}


def resolve_role(user_id: Optional[str]) -> str:
    """从 user_id 解析角色（演示：admin 前缀 / hr 前缀 / 其他为普通用户 / 空为 anonymous）"""
    if not user_id or user_id in ("unknown", "anonymous"):
        return "anonymous"
    uid = user_id.lower()
    if uid.startswith("admin"):
        return "admin"
    if uid.startswith("hr"):
        return "hr"
    return "user"


def check_tool_authz(tool_name: str, user_id: str, owner_hint: Optional[str] = None) -> tuple:
    """校验工具访问是否越权

    Args:
        tool_name: 工具名
        user_id: 当前用户 ID（thread config 传入）
        owner_hint: 数据归属提示（如申请人/客户名，用于归属校验）

    Returns:
        (allowed, reason): allowed 为 bool，reason 为拒绝原因（允许时为 ""）
    """
    role = resolve_role(user_id)
    allowed_roles = TOOL_ROLE_PERMISSIONS.get(tool_name)

    # 未注册工具：保守放行（不阻塞正常流程）
    if allowed_roles is None:
        return True, ""

    # 角色是否在允许集合
    if role not in allowed_roles:
        reason = f"工具 '{tool_name}' 需要角色 {'/'.join(allowed_roles - {'anonymous'}) if role == 'anonymous' else '/'.join(allowed_roles)}，当前角色 '{role}' 无权访问"
        logger.warning(f"[authz] 越权拦截: user={user_id!r} role={role} tool={tool_name} -> {reason}")
        return False, reason

    # 归属校验：普通用户仅能访问与自身相关的数据
    if role == "user" and owner_hint:
        # 简化演示：owner_hint 含当前 user_id 视为本人数据
        if owner_hint not in (user_id, ""):
            reason = f"数据归属 '{owner_hint}' 与当前用户 '{user_id}' 不匹配，无权访问他人数据"
            logger.warning(f"[authz] 归属校验拦截: {reason}")
            return False, reason

    return True, ""


def guard_tool_call(tool_name: str, user_id: str, owner_hint: Optional[str] = None,
                    config=None) -> tuple:
    """工具调用护栏入口（供 call_tools 包装器调用）

    Returns:
        (allowed, reason)
    """
    from utils.config import Config as C
    if config is None:
        config = C
    if not getattr(config, "ENABLE_GUARDRAILS", False):
        return True, ""
    return check_tool_authz(tool_name, user_id, owner_hint)
