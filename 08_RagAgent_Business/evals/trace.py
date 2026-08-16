# evals/trace.py
# 评测轨迹采集：从 graph.stream(stream_mode="updates") 事件中提取
# P1-A 可观测性扩展：记录节点执行时间线、每步工具、token 用量、延迟、重试次数
# 不侵入 MessagesState —— 图结构与状态零改动，仅从事件流与消息元数据采集

import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StepRecord:
    """一次图节点执行的观测记录"""
    node: str                                          # 节点名：agent/call_tools/grade_documents/rewrite/generate/...
    start_ts: float = 0.0                              # 节点开始执行的时间戳 (perf_counter)
    end_ts: float = 0.0                                # 节点结束的时间戳
    duration_ms: float = 0.0                           # 节点耗时（毫秒）
    tools: List[str] = field(default_factory=list)     # 该节点实际调用的工具名
    token_usage: dict = field(default_factory=dict)    # {"input": n, "output": n, "total": n}（若消息带 usage_metadata）
    message_type: str = ""                             # 产出消息类型 (AIMessage/HumanMessage/ToolMessage)
    has_tool_calls: bool = False                       # 是否产出工具调用


@dataclass
class Trace:
    """一次 graph 调用的完整轨迹"""
    case_id: str = ""
    # ---- 原有字段（向后兼容）----
    tool_messages: List[dict] = field(default_factory=list)   # [{"name": ..., "content": ...}] 按调用顺序
    final_answer: Optional[str] = None                         # generate 节点最终回答
    relevance_scores: List[str] = field(default_factory=list) # grade 节点的评分记录
    rewrite_count: int = 0
    strategy: Optional[str] = None                             # search_agent 选择的策略
    error: Optional[str] = None
    used_tools: List[str] = field(default_factory=list)        # 实际调用过的工具名（去重）

    # ---- P1-A 新增字段：可观测性 ----
    steps: List[StepRecord] = field(default_factory=list)       # 节点执行时间线（按顺序）
    plan_tree: List[dict] = field(default_factory=list)         # 规划树：[{"step": n, "action": ..., "status": "planned/done/failed", ...}]
    llm_calls: int = 0                                          # LLM 调用次数（由带 usage_metadata 的消息计数）
    total_tokens: int = 0                                       # token 总量（input+output）
    input_tokens: int = 0
    output_tokens: int = 0
    total_latency_ms: float = 0.0                              # 全链路延迟（首节点开始 → 末节点结束）
    retries: int = 0                                            # 工具重试总次数（P1-B 工具包装器回填）
    tool_retries: dict = field(default_factory=dict)            # 按工具名的重试次数 {"tool": n}

    def last_tool_output(self, tool_name: str) -> str:
        """返回最后一个同名工具的输出文本（hit 判定的数据源）"""
        for msg in reversed(self.tool_messages):
            if msg["name"] == tool_name:
                return msg["content"]
        return ""

    # ---------- P1-A 便捷统计 ----------
    @property
    def node_count(self) -> int:
        """执行的节点总数"""
        return len(self.steps)

    @property
    def effective_steps(self) -> int:
        """有效执行步数（跳过空/无消息节点）"""
        return len(self.steps)

    def tool_call_count(self) -> int:
        """工具调用总次数（含重复）"""
        return len(self.tool_messages)

    def tool_call_accuracy(self, expected_tools: List[str]) -> float:
        """工具调用准确率：期望工具是否都被调用（用于 Agent 行为层指标）
        返回 0.0~1.0：期望工具命中的比例
        """
        if not expected_tools:
            return 0.0
        used = set(self.used_tools)
        return len(set(expected_tools) & used) / len(set(expected_tools))

    def is_self_contained(self) -> bool:
        """是否为无效 LLM 调用（未调用任何工具却走到 generate）"""
        return len(self.tool_messages) == 0

    def to_dict(self) -> dict:
        """序列化为字典，供评测报告导出"""
        return {
            "case_id": self.case_id,
            "tool_messages": self.tool_messages,
            "final_answer": self.final_answer,
            "relevance_scores": self.relevance_scores,
            "rewrite_count": self.rewrite_count,
            "strategy": self.strategy,
            "error": self.error,
            "used_tools": self.used_tools,
            "steps": [
                {
                    "node": s.node,
                    "duration_ms": round(s.duration_ms, 2),
                    "tools": s.tools,
                    "token_usage": s.token_usage,
                    "message_type": s.message_type,
                    "has_tool_calls": s.has_tool_calls,
                }
                for s in self.steps
            ],
            "plan_tree": self.plan_tree,
            "llm_calls": self.llm_calls,
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "retries": self.retries,
            "tool_retries": self.tool_retries,
        }


def _message_content(content) -> str:
    """消息内容可能是 str 或 list[content_block]，统一转 str"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _extract_usage(msg) -> dict:
    """从消息中提取 token 用量（兼容 usage_metadata 与 response_metadata.usage）"""
    usage = {}
    um = getattr(msg, "usage_metadata", None)
    if isinstance(um, dict):
        usage["input"] = um.get("input_tokens", 0) or 0
        usage["output"] = um.get("output_tokens", 0) or 0
        usage["total"] = um.get("total_tokens", 0) or (usage["input"] + usage["output"])
        return usage
    rm = getattr(msg, "response_metadata", None) or {}
    tok = rm.get("token_usage") or rm.get("usage") or {}
    if isinstance(tok, dict):
        usage["input"] = tok.get("prompt_tokens") or tok.get("input_tokens") or 0
        usage["output"] = tok.get("completion_tokens") or tok.get("output_tokens") or 0
        usage["total"] = tok.get("total_tokens") or (usage["input"] + usage["output"])
    return usage


class _TimedStep:
    """采集过程中的当前节点计时器"""
    def __init__(self, node: str):
        self.record = StepRecord(node=node, start_ts=time.perf_counter())

    def finish(self) -> StepRecord:
        self.record.end_ts = time.perf_counter()
        self.record.duration_ms = (self.record.end_ts - self.record.start_ts) * 1000.0
        return self.record


def collect_trace(graph, case_id: str, query: str, config: dict,
                  trace: Optional["Trace"] = None,
                  plan_marker: Optional[dict] = None) -> Trace:
    """运行 graph 并采集完整轨迹

    Args:
        graph: 编译后的 LangGraph
        case_id: 用例 ID
        query: 用户问题
        config: 运行时 config（含 thread_id）
        trace: 可选的外部 Trace 实例（用于多轮累积采集）
        plan_marker: 可选的外部 plan 采集回调（Step2 规划树由 planner 节点回填，经此传出）

    Returns:
        Trace: 采集到的轨迹
    """
    trace = trace or Trace(case_id=case_id)
    _t0 = time.perf_counter()
    current_step: Optional[_TimedStep] = None
    node_tools: List[str] = []
    node_usage: dict = {"input": 0, "output": 0, "total": 0}

    def _flush_step():
        nonlocal current_step, node_tools, node_usage
        if current_step is not None:
            rec = current_step.finish()
            rec.tools = list(node_tools)
            if node_usage.get("total"):
                rec.token_usage = dict(node_usage)
                trace.input_tokens += node_usage.get("input", 0)
                trace.output_tokens += node_usage.get("output", 0)
                trace.total_tokens += node_usage.get("total", 0)
                trace.llm_calls += 1
            trace.steps.append(rec)
        current_step = None
        node_tools = []
        node_usage = {"input": 0, "output": 0, "total": 0}

    try:
        events = graph.stream(
            {"messages": [{"role": "user", "content": query}], "rewrite_count": 0},
            config,
            stream_mode="updates",
        )
        for event in events:
            for node_name, update in event.items():
                _flush_step()  # 进入新节点，结算上一个节点
                current_step = _TimedStep(node_name)

                # ---- 工具执行节点 ----
                if node_name == "call_tools":
                    for msg in update.get("messages", []):
                        name = getattr(msg, "name", "")
                        trace.tool_messages.append({
                            "name": name,
                            "content": _message_content(getattr(msg, "content", "")),
                        })
                        if name:
                            node_tools.append(name)
                            if name not in trace.used_tools:
                                trace.used_tools.append(name)
                # ---- 评分节点 ----
                elif node_name == "grade_documents":
                    score = update.get("relevance_score")
                    if score:
                        trace.relevance_scores.append(str(score))
                    # 评分节点内部有 LLM 调用，但 update 不直接携带消息 → 无法精确计量，
                    # 此处若带 messages 则尝试提取 usage
                    for msg in update.get("messages", []):
                        u = _extract_usage(msg)
                        if u.get("total"):
                            for k, v in u.items():
                                node_usage[k] = node_usage.get(k, 0) + v
                # ---- 生成节点 ----
                elif node_name == "generate":
                    for msg in update.get("messages", []):
                        content = _message_content(getattr(msg, "content", ""))
                        if content:
                            trace.final_answer = content
                        u = _extract_usage(msg)
                        if u.get("total"):
                            for k, v in u.items():
                                node_usage[k] = node_usage.get(k, 0) + v
                # ---- 重写节点 ----
                elif node_name == "rewrite":
                    trace.rewrite_count = update.get("rewrite_count", trace.rewrite_count)
                    for msg in update.get("messages", []):
                        u = _extract_usage(msg)
                        if u.get("total"):
                            for k, v in u.items():
                                node_usage[k] = node_usage.get(k, 0) + v
                # ---- agent 节点（LLM 决策，可能带工具调用）----
                elif node_name == "agent":
                    for msg in update.get("messages", []):
                        tc = getattr(msg, "tool_calls", None)
                        if tc:
                            current_step.record.has_tool_calls = True
                            for t in tc:
                                tname = t.get("name") if isinstance(t, dict) else getattr(t, "name", None)
                                if tname:
                                    node_tools.append(tname)
                        u = _extract_usage(msg)
                        if u.get("total"):
                            for k, v in u.items():
                                node_usage[k] = node_usage.get(k, 0) + v
                # ---- RL 搜索 agent ----
                elif node_name == "search_agent":
                    strategy = update.get("selected_strategy")
                    if strategy:
                        trace.strategy = strategy
                # ---- 规划节点（P0-A）----
                elif node_name == "planner":
                    if plan_marker is not None:
                        plan_marker.update({"planner_update": update})
                    # 若规划节点把 plan 写进 state，可经 update 读取
                    if "plan" in update:
                        trace.plan_tree = list(update.get("plan", []))

        _flush_step()  # 结算最后一个节点
        trace.total_latency_ms = (time.perf_counter() - _t0) * 1000.0
    except Exception as e:
        trace.error = str(e)
    return trace
