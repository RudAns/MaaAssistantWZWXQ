"""对局记录器：记录 (state, decision, result)，供 Laya 微调使用。

输出 JSONL 格式，每行一次决策点；对局结束时补写 final_rank。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional


class GameRecorder:
    def __init__(self, log_dir: Path | str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._path: Optional[Path] = None

    def start_game(self) -> Path:
        self._path = self.log_dir / f"game_{int(time.time())}.jsonl"
        self._file = open(self._path, "a", encoding="utf-8")
        return self._path

    def record_decision(
        self,
        state_dict: dict[str, Any],
        actions: list[dict[str, Any]],
        diagnostics: dict[str, Any],
    ) -> None:
        if self._file is None:
            self.start_game()
        record = {
            "ts": time.time(),
            "type": "decision",
            "state": state_dict,
            "actions": actions,
            "diagnostics": diagnostics,
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def record_result(self, final_rank: int, extra: Optional[dict] = None) -> None:
        """对局结束：记录最终名次，作为微调时的 reward 信号来源。"""
        if self._file is None:
            return
        record = {
            "ts": time.time(),
            "type": "result",
            "final_rank": final_rank,
            "extra": extra or {},
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
