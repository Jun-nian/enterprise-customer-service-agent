# evals/observability.py
# 可观测层（P1-A）：基于 LangChain 回调的 LLM 调用观测
# 精确捕获每次 LLM 调用的 token 用量、延迟、重试，作为 Agent 行为层评测的数据源
# 通过 with_observability(llm) 包裹 LLM，对工作流零侵入（仅挂回调）

import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMCallRecord:
    """一次 LLM 调用的观测记录"""
    index: int = 0
    node: str = "unknown"            # 触发的图节点（若可从回调链推断）
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_ms: float = 0.0
    retries: int = 0                 # 本次调用内部重试次数
    success: bool = True
    error: str = ""


class LLMObserver:
    """LLM 调用观测器：注册到 ChatOpenAI 的 callbacks，累积统计

    线程安全（评测为单线程跑批）；通过 start_node/end_node 标记当前图节点，
    以将 token 成本归因到具体节点（对评测的"每步成本"分析很有价值）。
    """

    def __init__(self):
        self.records: List[LLMCallRecord] = []
        self._active_node = "unknown"
        self._node_stack: List[str] = []
        # 用于重试观测：LangChain 每次重试会重新触发 on_llm_start
        self._attempt_counters = defaultdict(int)
        self._start_times: List[float] = []
        self._lock_owner = False

    # ---------- 节点归因 ----------
    @contextmanager
    def node_context(self, node: str):
        """上下文管理器：标记当前图节点，供评测 runner 在节点边界调用"""
        self._node_stack.append(node)
        self._active_node = node
        try:
            yield
        finally:
            if self._node_stack:
                self._node_stack.pop()
            self._active_node = self._node_stack[-1] if self._node_stack else "unknown"

    # ---------- 统计 ----------
    @property
    def total_calls(self) -> int:
        return len(self.records)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.records)

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.records)

    @property
    def total_latency_ms(self) -> float:
        return sum(r.duration_ms for r in self.records)

    @property
    def total_retries(self) -> int:
        return sum(r.retries for r in self.records)

    def by_node(self) -> Dict[str, dict]:
        """按节点聚合：{"node": {"calls": n, "tokens": n, "latency_ms": f}}"""
        agg: Dict[str, dict] = defaultdict(lambda: {"calls": 0, "tokens": 0, "latency_ms": 0.0, "retries": 0})
        for r in self.records:
            a = agg[r.node]
            a["calls"] += 1
            a["tokens"] += r.total_tokens
            a["latency_ms"] += r.duration_ms
            a["retries"] += r.retries
        return dict(agg)

    def reset(self):
        self.records.clear()
        self._attempt_counters.clear()
        self._start_times.clear()

    # ---------- 兼容简单调用 ----------
    def start_node(self, node: str):
        """标记当前节点（同步调用方式）"""
        self._node_stack.append(node)
        self._active_node = node

    def end_node(self):
        """结束当前节点标记"""
        if self._node_stack:
            self._node_stack.pop()
        self._active_node = self._node_stack[-1] if self._node_stack else "unknown"


def with_observability(llm_chat, observer: Optional[LLMObserver] = None):
    """给 ChatOpenAI 挂载观测回调，返回 (带回调的 llm, observer)

    Args:
        llm_chat: 原始 ChatOpenAI
        observer: 可选的已有 observer（复用累积）

    Returns:
        (llm, observer)：observer 始终可用
    """
    obs = observer or LLMObserver()

    from langchain_core.callbacks import BaseCallbackHandler

    class _ObserverHandler(BaseCallbackHandler):
        def __init__(self, target: LLMObserver):
            self.target = target
            self._pending: Dict[int, dict] = {}   # run_id -> {"t0":..,"node":..,"attempts":..}

        def on_llm_start(self, serialized, prompts, **kwargs):
            run_id = kwargs.get("run_id")
            if run_id is None:
                run_id = len(self.target.records) + len(self._pending) + 1
            self._pending[run_id] = {
                "t0": time.perf_counter(),
                "node": self.target._active_node,
                "attempts": 0,
            }

        def on_llm_error(self, error, **kwargs):
            run_id = kwargs.get("run_id")
            if run_id in self._pending:
                self._pending[run_id]["attempts"] += 1  # 失败会触发重试 → 计为一次重试
                # 若为最终失败，记录一次 error
                self._pending[run_id]["error"] = str(error)

        def on_llm_end(self, response, **kwargs):
            run_id = kwargs.get("run_id")
            if run_id not in self._pending:
                return
            p = self._pending.pop(run_id)
            dur_ms = (time.perf_counter() - p["t0"]) * 1000.0
            # 提取 token 用量（OpenAI 风格）
            pt, ct = 0, 0
            try:
                llm_output = getattr(response, "llm_output", None) or {}
                token_usage = llm_output.get("token_usage") or {}
                pt = token_usage.get("prompt_tokens", 0) or 0
                ct = token_usage.get("completion_tokens", 0) or 0
            except Exception:
                pass
            rec = LLMCallRecord(
                index=len(self.target.records),
                node=p["node"],
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=pt + ct,
                duration_ms=dur_ms,
                retries=p.get("attempts", 0),
                success=not p.get("error"),
                error=p.get("error", ""),
            )
            self.target.records.append(rec)

    handler = _ObserverHandler(obs)
    llm_chat.callbacks = [handler]
    return llm_chat, obs


def summarize_observer(observer: LLMObserver) -> dict:
    """将 observer 统计汇总为评测指标所需的字典"""
    return {
        "llm_calls": observer.total_calls,
        "prompt_tokens": observer.total_prompt_tokens,
        "completion_tokens": observer.total_completion_tokens,
        "total_tokens": observer.total_tokens,
        "total_latency_ms": round(observer.total_latency_ms, 2),
        "total_retries": observer.total_retries,
        "by_node": {k: {"calls": v["calls"], "tokens": v["tokens"],
                         "latency_ms": round(v["latency_ms"], 2), "retries": v["retries"]}
                    for k, v in observer.by_node().items()},
    }
