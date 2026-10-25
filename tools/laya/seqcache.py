"""预分词：把每道题的"题干 / 每个选项 / 局面"各分词一次缓存起来，训练时按选项顺序直接拼。

与 laya.common.build_sequence **逐 token 等价**（包括选项打乱、选项超长截断、题干截断、局面截断的全部规则），
只是省掉每一步重复分词。等价性由 `python tools/laya/seqcache.py --check` 在全部决策点 × 多种随机顺序上逐一比对验证。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from laya.common import build_sequence, render_options, serialize_state


class SeqCache:
    def __init__(self, tok, max_len: int, head_max_len: int):
        self.tok, self.max_len, self.head_max_len = tok, max_len, head_max_len
        self._c: dict = {}

    def _pieces(self, key: str, state, q: dict):
        c = self._c.get(key)
        if c is None:
            tok, mask_tok = self.tok, self.tok.mask_token
            opts = render_options(q)
            ins = str(q["ins"]).replace(mask_tok, " ")
            head = tok("%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
            opt_ids = [[tok.mask_token_id] + tok(" " + o.replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:48]
                       for o in opts]
            st = tok(serialize_state(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
            c = self._c[key] = (head, opt_ids, st)
        return c

    def build(self, key: str, state, q: dict, option_order=None):
        """与 build_sequence(tok, state, q, max_len, head_max_len, option_order) 的返回值相同。"""
        head_ids, all_opts, st = self._pieces(key, state, q)
        order = option_order if option_order is not None else list(range(len(all_opts)))
        opt_ids = [all_opts[i] for i in order]
        hm, ml, tok = self.head_max_len, self.max_len, self.tok
        opt_budget = hm - sum(len(o) for o in opt_ids)
        if opt_budget < 16:
            per = max(4, (hm - 16) // max(1, len(opt_ids)))
            opt_ids = [o[:per] for o in opt_ids]
            opt_budget = hm - sum(len(o) for o in opt_ids)
        head = head_ids[: max(8, opt_budget)]
        ids = [tok.cls_token_id] + head + [tok.sep_token_id]
        markers = []
        for o in opt_ids:
            markers.append(len(ids))
            ids.extend(o)
        ids.append(tok.sep_token_id)
        room = max(0, ml - len(ids) - 1)
        ids = ids + st[:room] + [tok.sep_token_id]
        return ids[:ml], [m for m in markers if m < ml]


def _check(ver: str, reps: int) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import features as F
    from train import BASE_CKPT
    from transformers import AutoTokenizer
    cfg = json.loads((BASE_CKPT / "rl_agent_config.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(str(BASE_CKPT / "tokenizer"))
    cache = SeqCache(tok, cfg["max_len"], cfg["head_max_len"])
    root = Path(__file__).resolve().parents[2]
    rng = random.Random(0)
    n = bad = 0
    for line in (root / "data" / "laya" / "datasets" / ver / "points.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        q = F.to_internal(p["question"])
        k = len(q["crit"])
        for r in range(reps):
            order = None if r == 0 else rng.sample(range(k), k)
            want = build_sequence(tok, p["state"], q, cfg["max_len"], cfg["head_max_len"], option_order=order)
            got = cache.build(p["id"], p["state"], q, option_order=order)
            n += 1
            if tuple(map(tuple, want)) != tuple(map(tuple, got)):
                bad += 1
                if bad <= 3:
                    print("不一致：", p["id"], order)
    print(f"{ver}: 比对 {n} 个序列，不一致 {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ver", default="v4")
    ap.add_argument("--reps", type=int, default=4)
    a = ap.parse_args()
    sys.exit(_check(a.ver, a.reps) if a.check else 0)
