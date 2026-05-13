import os
import json
from typing import Any

from utils import now_str


class StrategyEngine:

    def __init__(self):

        # 默认策略目录（相对程序运行目录）
        self.strategy_dir = "strategies"

        os.makedirs(
            self.strategy_dir,
            exist_ok=True
        )

    # =====================================
    # 动量轮动策略
    # =====================================

    def momentum_rotation_strategy(
            self,
            top_n=3,
            factor="roc12"
    ):
        return {
            "version": 1,
            "name": f"Momentum_{top_n}",
            "type": "rule_based",
            "created_at": now_str(),
            "description": "选择动量最强ETF轮动",
            "buy_rule": {
                "factor": factor,
                "top_n": int(top_n),
                "ascending": False
            },
            "sell_rule": {
                "rebalance": {"enabled": True},
            },
            "execution": {
                "decision_at": "close",
                "buy_at": "open_next",
                "sell_at": "open",
                "min_hold_days": 1
            },
            "position": {
                "max_positions": int(top_n)
            }
        }

    # =====================================
    # 多因子加权（横截面）
    # =====================================

    def composite_factor_strategy(
        self,
        *,
        name: str,
        factors: list[dict[str, Any]],
        top_n: int = 3,
    ):
        """
        factors: [{"name": "roc12", "weight": 1.0}, ...]
        """
        return {
            "version": 1,
            "name": name,
            "type": "composite_factor",
            "created_at": now_str(),
            "description": "多因子加权横截面选股",
            "buy_rule": {
                "factors": factors,
                "top_n": int(top_n),
            },
            "sell_rule": {
                "rebalance": {"enabled": True},
            },
            "execution": {
                "decision_at": "close",
                "buy_at": "open_next",
                "sell_at": "open",
                "min_hold_days": 1
            },
            "position": {
                "max_positions": int(top_n)
            },
        }

    # =====================================
    # 保存策略
    # =====================================

    def save_strategy(

            self,
            strategy
    ):

        name = strategy["name"]

        path = os.path.join(self.strategy_dir, f"{name}.json")

        with open(
                path,
                "w",
                encoding="utf-8"
        ) as f:

            json.dump(

                strategy,

                f,

                indent=4,

                ensure_ascii=False
            )

        return path

    def save_strategy_as(self, strategy: dict, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(strategy, f, indent=4, ensure_ascii=False)
        return path

    # =====================================
    # 加载策略
    # =====================================

    def load_strategy(self, name):

        path = os.path.join(

            self.strategy_dir,

            f"{name}.json"
        )

        with open(
                path,
                "r",
                encoding="utf-8"
        ) as f:

            return json.load(f)

    def load_strategy_file(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # =====================================
    # 获取全部策略
    # =====================================

    def get_all_strategies(self):

        result = []

        for fn in os.listdir(self.strategy_dir):

            if fn.endswith(".json"):

                result.append(
                    fn.replace(".json", "")
                )

        return result

    # =====================================
    # 导入策略（拷贝进默认目录）
    # =====================================

    def import_strategy(self, path: str):
        stg = self.load_strategy_file(path)
        name = stg.get("name") or os.path.splitext(os.path.basename(path))[0]
        stg["name"] = name
        self.save_strategy(stg)
        return name
