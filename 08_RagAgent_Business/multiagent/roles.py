# multiagent/roles.py
# P1-C 多智能体协作：Planner / Executor / Critic 角色化 + 领域子 Agent 路由
# 让"多 Agent"从简历文字变为代码事实：
#   - 角色定义（AgentRole）：每个角色有独立名称、职责、系统提示词
#   - 角色路由（RoleRouter）：根据问题意图路由到领域子 Agent
#   - 领域子 Agent：HR / IT / 业务 / 知识库，各自持有工具子集
#
# 面试话术钩子：
#   "将单 Agent 拆分为 Planner/Executor/Critic 角色 + 领域子 Agent 路由，
#    呼应腾讯 NDR 项目'四层可扩展 Agent（网关→编排→插件→内核）'架构。"

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------- 角色定义 ----------

@dataclass
class AgentRole:
    """智能体角色元数据"""
    name: str
    title: str                    # 中文标题
    responsibility: str           # 职责描述
    system_hint: str = ""         # 角色化系统提示词片段


# 规划三元组角色（与 planning/ 模块对应）
ROLE_PLANNER = AgentRole(
    name="planner",
    title="规划智能体",
    responsibility="分析用户问题，拆解为多步执行计划，判定简单/复杂",
    system_hint="你是一名任务规划专家，负责把复杂问题拆解为可执行步骤。",
)
ROLE_EXECUTOR = AgentRole(
    name="executor",
    title="执行智能体",
    responsibility="按计划逐步调用工具执行，收集观察结果",
    system_hint="你是一名执行专家，严格按计划调用工具完成任务。",
)
ROLE_CRITIC = AgentRole(
    name="critic",
    title="评估智能体",
    responsibility="评估执行结果是否满足目标，决定继续/完成/重规划",
    system_hint="你是一名质量评估专家，判断结果是否达成目标。",
)

# 领域子 Agent（业务域划分）
DOMAIN_AGENTS = {
    "hr": AgentRole(
        name="hr_agent",
        title="HR 智能体",
        responsibility="处理入职/考勤/报销/HR政策/员工福利等 HR 领域问题",
        system_hint="你是企业 HR 领域的专属智能体，擅长员工与人事政策。",
    ),
    "it": AgentRole(
        name="it_agent",
        title="IT 智能体",
        responsibility="处理 IT 支持、工单、设备故障等 IT 领域问题",
        system_hint="你是企业 IT 领域的专属智能体，擅长技术支持与工单。",
    ),
    "business": AgentRole(
        name="business_agent",
        title="业务智能体",
        responsibility="处理销售订单、客户、交付等业务领域问题",
        system_hint="你是企业业务领域的专属智能体，擅长订单与客户。",
    ),
    "knowledge": AgentRole(
        name="knowledge_agent",
        title="知识库智能体",
        responsibility="检索企业知识库（制度/产品文档/技术方案）",
        system_hint="你是企业知识库专属智能体，擅长语义检索。",
    ),
}


# ---------- 领域路由规则 ----------

# 意图关键词 → 领域 Agent
_ROUTING_RULES = [
    ("hr", ["入职", "离职", "考勤", "请假", "年假", "报销", "工资", "社保", "公积金", "福利", "招聘", "hr", "人事", "加班"]),
    ("it", ["it", "工单", "电脑", "网络", "wifi", "vpn", "软件", "系统故障", "报修", "邮箱", "密码", "设备"]),
    ("business", ["订单", "客户", "销售", "交付", "合同", "采购", "报价", "业务", "发货"]),
    ("knowledge", ["制度", "手册", "产品", "技术", "方案", "规范", "流程", "政策"]),
]


def route_to_domain(query: str) -> str:
    """根据查询意图路由到领域子 Agent（返回 agent 名）

    Args:
        query: 用户问题

    Returns:
        str: 领域 Agent 名（hr_agent / it_agent / business_agent / knowledge_agent）
    """
    q = (query or "").lower()
    for domain, keywords in _ROUTING_RULES:
        for kw in keywords:
            if kw.lower() in q:
                logger.info(f"[role_router] '{query}' 命中领域关键词 '{kw}' → {domain}_agent")
                return f"{domain}_agent"
    # 兜底：知识库
    return "knowledge_agent"


def get_domain_tools(domain_agent_name: str, all_tools) -> List:
    """根据领域 Agent 筛选可用工具子集

    Args:
        domain_agent_name: 领域 Agent 名（hr_agent 等）
        all_tools: 全部工具列表

    Returns:
        list: 该领域 Agent 允许使用的工具
    """
    domain = domain_agent_name.replace("_agent", "")
    tool_by_domain = {
        "hr": {"search_employees", "search_faq"},
        "it": {"search_tickets", "search_faq"},
        "business": {"search_orders", "search_faq"},
        "knowledge": {"retrieve"},
    }
    allowed = tool_by_domain.get(domain, set())
    return [t for t in all_tools if t.name in allowed]


# ---------- 角色化执行上下文（供工作流节点标注角色） ----------

@dataclass
class AgentContext:
    """一次对话中的多智能体执行上下文"""
    current_role: str = "agent"                 # 当前执行的角色（planner/executor/critic）
    domain_agent: str = ""                       # 路由到的领域子 Agent
    role_trace: List[str] = field(default_factory=list)  # 角色执行轨迹

    def enter(self, role: str):
        self.current_role = role
        self.role_trace.append(f"{role}@{len(self.role_trace)}")

    def __repr__(self):
        return f"<AgentContext role={self.current_role} domain={self.domain_agent} trace={self.role_trace}>"


def build_agent_context(query: str) -> AgentContext:
    """构建多智能体上下文（路由领域 + 记录角色轨迹）"""
    ctx = AgentContext()
    ctx.domain_agent = route_to_domain(query)
    return ctx


# 供测试/调试
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    tests = [
        "公司的入职流程是什么",
        "我的电脑坏了，查一下工单进度",
        "查一下客户的订单状态",
        "公司的产品技术方案有哪些",
    ]
    for t in tests:
        print(f"  {t[:20]:<22} → {route_to_domain(t)}")
