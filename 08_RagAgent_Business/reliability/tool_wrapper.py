# reliability/tool_wrapper.py
# P1-B 工具可靠性：超时 / 重试（指数退避）/ 失败降级 / 熔断
# 对 LangGraph 工具调用统一包装，防止单个故障工具拖垮整个 Agent。
#
# 面试话术钩子：
#   "工具层四重防护：超时熔断故障工具、指数退避重试瞬时错误、失败降级返回兜底、
#    熔断器快速失败避免雪崩——呼应腾讯 NDR 项目'8 并发断点续跑/双实例稳定'叙事。"

import functools
import logging
import random
import threading
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """熔断器：连续失败超过阈值 → 打开（快速失败），冷却后半开试探，成功则关闭"""

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 30.0):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._lock = threading.Lock()
        self.state = "closed"   # closed / open / half_open

    def allow(self) -> bool:
        """是否允许调用（熔断器状态检查）"""
        with self._lock:
            if self.state == "open":
                # 冷却期结束 → 半开试探
                if time.time() - (self._opened_at or 0) >= self.cooldown_seconds:
                    self.state = "half_open"
                    return True
                return False
            return True

    def record_success(self):
        """调用成功 → 重置计数，关闭熔断器"""
        with self._lock:
            self._failures = 0
            self.state = "closed"
            self._opened_at = None

    def record_failure(self):
        """调用失败 → 累计，超过阈值打开熔断器"""
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self.state = "open"
                self._opened_at = time.time()
                logger.warning(f"[circuit_breaker] 连续 {self._failures} 次失败，熔断器打开（冷却 {self.cooldown_seconds}s）")
            elif self.state == "half_open":
                # 半开试探失败 → 立即重新打开
                self.state = "open"
                self._opened_at = time.time()


class ToolReliabilityConfig:
    """工具可靠性配置"""
    def __init__(self, timeout_seconds: float = 20.0,
                 max_retries: int = 2,
                 backoff_base: float = 1.0,
                 backoff_factor: float = 2.0,
                 breaker_threshold: int = 3,
                 breaker_cooldown: float = 30.0,
                 fallback_text: str = "工具调用暂时不可用，请稍后重试或联系企业支持。"):
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_factor = backoff_factor
        self.breaker_threshold = breaker_threshold
        self.breaker_cooldown = breaker_cooldown
        self.fallback_text = fallback_text


# 全局熔断器表（按工具名隔离，避免一个故障工具影响其他工具）
_BREAKERS: Dict[str, CircuitBreaker] = {}
_BREAKERS_LOCK = threading.Lock()


def _get_breaker(tool_name: str, cfg: ToolReliabilityConfig) -> CircuitBreaker:
    with _BREAKERS_LOCK:
        if tool_name not in _BREAKERS:
            _BREAKERS[tool_name] = CircuitBreaker(
                failure_threshold=cfg.breaker_threshold,
                cooldown_seconds=cfg.breaker_cooldown,
            )
        return _BREAKERS[tool_name]


def _execute_with_timeout(func: Callable, timeout: float, *args, **kwargs):
    """带超时的执行（线程内运行，超时抛 TimeoutError）"""
    result_holder = {}
    error_holder = {}

    def _run():
        try:
            result_holder["result"] = func(*args, **kwargs)
        except Exception as e:
            error_holder["error"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"工具执行超时（>{timeout}s）")
    if "error" in error_holder:
        raise error_holder["error"]
    return result_holder.get("result")


def wrap_tool(tool, config: Optional[ToolReliabilityConfig] = None, user_id: str = ""):
    """包装 LangChain 工具，注入可靠性防护

    Args:
        tool: 原始工具（LangChain @tool 或自定义可调用对象）
        config: 可靠性配置
        user_id: 当前用户（透传给 AuthZ）

    Returns:
        包装后的工具（保留 name / description / args_schema 供 bind_tools 使用）
    """
    cfg = config or ToolReliabilityConfig()
    tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", "unknown")

    @functools.wraps(tool)
    def wrapped(*args, **kwargs):
        breaker = _get_breaker(tool_name, cfg)

        # 熔断检查
        if not breaker.allow():
            logger.warning(f"[reliability] 熔断器打开，工具 {tool_name} 快速失败")
            return _fallback(tool, cfg, reason="circuit_open")

        # 工具执行体：兼容 BaseTool.invoke(args_dict) 与普通可调用对象
        def _invoke_impl():
            has_invoke = callable(getattr(tool, "invoke", None)) and not callable(tool)
            if has_invoke:
                if kwargs:
                    return tool.invoke(kwargs)
                if args and isinstance(args[0], dict):
                    return tool.invoke(args[0])
                return tool.invoke(args[0] if args else None)
            # 普通函数直接调用
            if kwargs:
                return tool(**kwargs)
            if args and isinstance(args[0], dict):
                return tool(**args[0])
            return tool(*args)

        attempts = 0
        delay = cfg.backoff_base
        while True:
            try:
                result = _execute_with_timeout(_invoke_impl, cfg.timeout_seconds)
                breaker.record_success()
                return result
            except TimeoutError:
                breaker.record_failure()
                attempts += 1
                if attempts > cfg.max_retries:
                    logger.error(f"[reliability] 工具 {tool_name} 超时（{cfg.timeout_seconds}s），重试 {attempts} 次后放弃")
                    return _fallback(tool, cfg, reason="timeout")
                logger.warning(f"[reliability] 工具 {tool_name} 超时，第 {attempts} 次重试（退避 {delay:.1f}s）")
            except Exception as e:
                breaker.record_failure()
                attempts += 1
                if attempts > cfg.max_retries:
                    logger.error(f"[reliability] 工具 {tool_name} 失败: {e}，重试 {attempts} 次后放弃")
                    return _fallback(tool, cfg, reason=f"error: {str(e)[:80]}")
                logger.warning(f"[reliability] 工具 {tool_name} 异常: {e}，第 {attempts} 次重试（退避 {delay:.1f}s）")
            time.sleep(delay + random.uniform(0, 0.5))
            delay *= cfg.backoff_factor

    # 保留工具元数据，供 LangChain bind_tools / ToolNode 使用
    wrapped.name = tool_name
    wrapped.description = getattr(tool, "description", "")
    if hasattr(tool, "args_schema"):
        wrapped.args_schema = tool.args_schema
    # 兼容 .invoke(args_dict) 调用方式（LangChain 工具标准接口）
    def _invoke(args: dict) -> Any:
        if isinstance(args, dict):
            return wrapped(**args)
        return wrapped(args)
    wrapped.invoke = _invoke
    return wrapped


def _fallback(tool, cfg: ToolReliabilityConfig, reason: str) -> str:
    """失败降级：返回兜底文本（而不是抛出异常导致 Agent 中断）"""
    text = f"工具 {getattr(tool, 'name', '')} 调用失败（{reason}）。{cfg.fallback_text}"
    return text


def reset_circuit_breakers():
    """重置全部熔断器（测试用）"""
    with _BREAKERS_LOCK:
        _BREAKERS.clear()
