# rl/bandit.py
# epsilon-greedy 多臂老虎机：为检索策略选择提供在线学习
# 奖励 = grade_documents 的 binary_score（yes=1.0 / no=0.0），1 轮延迟回填
# 状态持久化到 JSON 文件（跨进程累积）

import json
import os
import random

STRATEGIES = ["default", "multi_query", "top_k"]


class EpsilonGreedyBandit:
    """epsilon-greedy bandit，支持探索率衰减"""

    def __init__(self, strategies: list = None, epsilon: float = 0.1, decay: float = 0.99,
                 epsilon_min: float = 0.01, seed: int = None):
        """
        Args:
            strategies: 策略列表（默认 STRATEGIES）
            epsilon: 初始探索率
            decay: 每轮衰减系数
            epsilon_min: 探索率下限
            seed: 随机种子（评测可复现）
        """
        self.strategies = strategies or list(STRATEGIES)
        self.epsilon = epsilon
        self.decay = decay
        self.epsilon_min = epsilon_min
        self.counts = {s: 0 for s in self.strategies}       # 每个策略被选择的次数
        self.rewards = {s: 0.0 for s in self.strategies}    # 每个策略累计奖励
        self.rounds = 0
        self._rng = random.Random(seed)

    # ---------- 决策 ----------
    def choose(self, force: str = None) -> str:
        """选择一个策略

        Args:
            force: 若指定，直接返回该策略（离线评测固定策略用）

        Returns:
            str: 策略名
        """
        if force and force in self.strategies:
            return force
        self.rounds += 1
        self.epsilon = max(self.epsilon * self.decay, self.epsilon_min)
        if self._rng.random() < self.epsilon:
            # 探索：均匀随机选
            strategy = self._rng.choice(self.strategies)
        else:
            # 利用：选平均奖励最高的策略（未尝试过的策略优先）
            best = max(self.strategies, key=lambda s: (
                self.rewards[s] / self.counts[s] if self.counts[s] > 0 else float("inf"),
                self._rng.random(),
            ))
            strategy = best
        return strategy

    # ---------- 学习 ----------
    def update(self, strategy: str, reward: float):
        """回填奖励

        Args:
            strategy: 实际执行的策略名
            reward: 奖励（0.0~1.0）
        """
        if strategy not in self.counts:
            return
        self.counts[strategy] += 1
        self.rewards[strategy] += reward

    # ---------- 状态 ----------
    def state(self) -> dict:
        return {
            "strategies": self.strategies,
            "epsilon": self.epsilon,
            "rounds": self.rounds,
            "counts": self.counts,
            "rewards": self.rewards,
        }

    def summary(self) -> str:
        parts = [f"{s}: {self.counts[s]}次 均奖{self.rewards[s]/max(self.counts[s],1):.2f}"
                 for s in self.strategies]
        return " | ".join(parts)


def load_bandit(path: str) -> EpsilonGreedyBandit:
    """从 JSON 文件加载 bandit 状态（文件不存在则新建）"""
    bandit = EpsilonGreedyBandit()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            bandit.strategies = data.get("strategies", STRATEGIES)
            bandit.epsilon = data.get("epsilon", 0.1)
            bandit.rounds = data.get("rounds", 0)
            bandit.counts = {s: data.get("counts", {}).get(s, 0) for s in bandit.strategies}
            bandit.rewards = {s: data.get("rewards", {}).get(s, 0.0) for s in bandit.strategies}
        except Exception as e:
            print(f"[bandit] 加载 {path} 失败，使用新实例: {e}")
    return bandit


def save_bandit(bandit: EpsilonGreedyBandit, path: str):
    """保存 bandit 状态到 JSON 文件"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bandit.state(), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[bandit] 保存 {path} 失败: {e}")
