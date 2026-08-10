# evals/trace.py
# 评测轨迹采集：从 graph.stream(stream_mode="updates") 事件中提取
# 不侵入 MessagesState —— 图结构与状态零改动

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Trace:
    """一次 graph 调用的完整轨迹"""
    case_id: str = ""
    tool_messages: List[dict] = field(default_factory=list)   # [{"name": ..., "content": ...}] 按调用顺序
    final_answer: Optional[str] = None                         # generate 节点最终回答
    relevance_scores: List[str] = field(default_factory=list) # grade 节点的评分记录
    rewrite_count: int = 0
    strategy: Optional[str] = None                             # search_agent 选择的策略
    error: Optional[str] = None
    used_tools: List[str] = field(default_factory=list)        # 实际调用过的工具名（去重）

    def last_tool_output(self, tool_name: str) -> str:
        """返回最后一个同名工具的输出文本（hit 判定的数据源）"""
        for msg in reversed(self.tool_messages):
            if msg["name"] == tool_name:
                return msg["content"]
        return ""


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


def collect_trace(graph, case_id: str, query: str, config: dict) -> Trace:
    """运行 graph 并采集完整轨迹

    Args:
        graph: 编译后的 LangGraph
        case_id: 用例 ID
        query: 用户问题
        config: 运行时 config（含 thread_id）

    Returns:
        Trace: 采集到的轨迹
    """
    trace = Trace(case_id=case_id)
    try:
        # stream_mode="updates" 逐节点返回状态增量 {node_name: update_dict}
        events = graph.stream(
            {"messages": [{"role": "user", "content": query}], "rewrite_count": 0},
            config,
            stream_mode="updates",
        )
        for event in events:
            for node_name, update in event.items():
                if node_name == "call_tools":
                    for msg in update.get("messages", []):
                        name = getattr(msg, "name", "")
                        trace.tool_messages.append({
                            "name": name,
                            "content": _message_content(getattr(msg, "content", "")),
                        })
                        if name and name not in trace.used_tools:
                            trace.used_tools.append(name)
                elif node_name == "grade_documents":
                    score = update.get("relevance_score")
                    if score:
                        trace.relevance_scores.append(str(score))
                elif node_name == "generate":
                    for msg in update.get("messages", []):
                        content = _message_content(getattr(msg, "content", ""))
                        if content:
                            trace.final_answer = content
                elif node_name == "rewrite":
                    trace.rewrite_count = update.get("rewrite_count", trace.rewrite_count)
                elif node_name == "search_agent":
                    strategy = update.get("selected_strategy")
                    if strategy:
                        trace.strategy = strategy
    except Exception as e:
        trace.error = str(e)
    return trace
