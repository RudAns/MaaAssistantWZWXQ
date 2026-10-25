"""cgnet（局面判断网络）线上推理：接管 carry（谁当主C）/ swap（人口满卖谁腾位）两种题，其余题型仍用 Laya。

2026-09-30 分工：换主C / 定主C 与"卖谁换人"交给局面判断网络，其余继续 Laya。

用法（laya_policy 里接线）：
    from agent.decision.cgnet_policy import CGNetPolicy
    self.cg = CGNetPolicy.load(run_dir=run_dir)          # 进程启动时一次；失败返回 None（退回 Laya）
    ...
    # LayaPolicy._ask(p) 开头：
    if self.cg is not None and p["kind"] in self.cg.kinds:
        r = self.cg.ask(p)                               # -> (choice, probs, conf, ms) 或 None（出错 → 退回 Laya）
        if r is not None:
            return r

输入 p = build_dataset.carry_point / swap_point 造出来的同一道题（_restrict 删过选项也行）；
特征构造与训练共用 tools/cgnet/build_data.py 的 Builder.convert（v2 再加 build_data_v2.V2.augment），
"已得天赋"与训练一样从 run_dir/states.jsonl 回放（增量读，只读新增行）。
推理：onnxruntime CPU，多个种子的 ONNX 取概率平均。

一致性：tools/cgnet/test_live_parity.py（历史 carry/swap 题：线上路径 vs 训练路径，输入张量与输出概率逐项比对）。
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parents[2]
DEPLOY_DIR = PROJECT_DIR / "data" / "cgnet" / "deploy"          # export 产物：manifest.json + seed*.onnx + 数据表
NAMES = ["tt", "ot", "node", "text", "num", "glob", "mask", "rel", "opt", "optm"]


def collate_np(s: dict, n_ch: int) -> dict:
    """单题 → 模型输入（numpy）。与 tools/cgnet/train.collate 在 batch=1 时逐元素相同（test_live_parity 校验）。"""
    n = len(s["tt"])
    k = len(s["opt_pos"])
    rel = np.zeros((1, n, n, n_ch), np.float32)
    r = s["rel"]
    if len(r):
        rel[0, r[:, 0], r[:, 1], r[:, 2]] = s["relv"].astype(np.float32) if "relv" in s else 1.0
    return {"tt": s["tt"].astype(np.int64)[None], "ot": s["ot"].astype(np.int64)[None],
            "node": s["node"].astype(np.int64)[None], "text": s["text"].astype(np.int64)[None],
            "num": s["num"].astype(np.float32)[None], "glob": s["glob"].astype(np.float32)[None],
            "mask": np.ones((1, n), bool), "rel": rel, "opt": s["opt_pos"].astype(np.int64)[None],
            "optm": np.ones((1, k), bool)}


def apply_ablation(s: dict, man: dict) -> dict:
    """部署配置若是某个 v2 消融（manifest["v2ablate"]），线上特征做同样的处理（与 tools/cgnet/train_v2.apply 相同，
    test_live_parity 用 train_v2.apply 交叉校验）。"""
    mode = man.get("v2ablate") or "none"
    if mode == "none":
        return s
    cr = man["channel_ranges"]
    s = dict(s)
    s["num"], s["glob"] = s["num"].copy(), s["glob"].copy()
    r, rv = s["rel"], s["relv"]
    keep = np.ones(len(r), bool)
    if mode in ("no_facts", "no_v2"):
        s["glob"][man["fact_glob"][0]:man["fact_glob"][1]] = 0
        s["num"][:, man["fact_num"][0]:man["fact_num"][1]] = 0
    if mode in ("no_techdyn", "no_v2"):
        keep &= ~((r[:, 2] >= cr["slot"][0]) & (r[:, 2] < cr["slot"][1]))
        s["num"][:, man["tech_num"][0]:man["tech_num"][1]] = 0
    if mode in ("no_cond", "no_v2"):
        keep &= ~((r[:, 2] >= cr["cond"][0]) & (r[:, 2] < cr["cond"][1]))
    if mode == "no_tech":
        n = int((s["tt"] != man["tt_technique"]).sum())
        for k in ("tt", "ot", "node", "text", "num"):
            s[k] = s[k][:n]
        s["names"] = s["names"][:n]
        keep &= (r[:, 0] < n) & (r[:, 1] < n)
    s["rel"], s["relv"] = r[keep], rv[keep]
    return s



ENGINE_PKG = "agent.engine_v1"     # 已部署模型（cs_r9zd / R28）训练时的推演器冻结版；改推演器后要重新打分重训才能换成新版


def engine_annot(p: dict) -> dict:
    """线上现算推演器打分（与训练用的 tools/engine/annotate_points.py 同一函数、同样 n=8 组参数）。出错返回 {}（特征全 0，等于没有）。"""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "engine"))
        import annotate_points as _AP                 # noqa: E402
        r = _AP.annotate((json.dumps(p, ensure_ascii=False, default=str), 8, ENGINE_PKG)) or {}
        return r if r.get("opts") else {}
    except Exception:                                  # noqa: BLE001
        logger.exception("推演器打分失败（特征置 0）")
        return {}

class TalentTail:
    """增量回放 states.jsonl 里执行过的 talent_pick（与 build_data.replay_run 同一套规则：去重、新局清空）。"""

    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path else None
        self.pos = 0
        self.owned: list = []
        self.last_round = 0
        self.buf = ""

    def update(self) -> list:
        if not self.path or not self.path.exists():
            return list(self.owned)
        with self.path.open("r", encoding="utf-8") as f:
            f.seek(self.pos)
            chunk = f.read()
            self.pos = f.tell()
        data = self.buf + chunk
        lines = data.split("\n")
        self.buf = lines.pop()                       # 最后一行可能还没写完
        for line in lines:
            try:
                self.feed(json.loads(line))
            except ValueError:
                continue
        return list(self.owned)


    def feed(self, x: dict) -> None:
        """处理一帧记录（与 build_data.replay_run 的逐帧逻辑相同）。"""
        s = x.get("state") or {}
        if x.get("executed") and x.get("actions"):
            a = x["actions"][0]
            if a.get("type") == "talent_pick" and s.get("phase") == "talent":
                nm = a.get("name")
                if not nm:
                    tl = s.get("talents") or []
                    i = a.get("index", -1)
                    nm = tl[i]["name"] if 0 <= i < len(tl) else None
                nm = re.sub(r"\s+", "", nm or "")
                if nm and nm not in self.owned:
                    self.owned.append(nm)
        rn = s.get("round_no") or 0
        if s.get("phase") in ("lobby", "hero_pick") or (rn and rn < self.last_round - 1):
            self.owned = []
        if rn:
            self.last_round = rn


class CGNetModel:
    """一个部署包（一个配置、若干种子）。"""
    kinds = frozenset({"carry", "swap"})
    # 训练数据（r9m）里所有题的 ctx 都没有锁定路线（route 字段整列为空）→ 与路线有关的输入（路线成员标记 / 关系通道 /
    # 路线锁定位）在训练中恒为 0，权重没学过。线上若路线已锁定，照样喂进去会产生训练中从未见过的输入 → 这里屏蔽，
    # 与训练分布一致。等训练数据里有路线后再关掉。
    # 2026-10-11 关掉：r9s / r9t 起训练题约 80% 带路线，线上却屏蔽 → 输入分布反而不一致。
    # 部署包 cs_r9t 在 r9t_cg 测试集上：不屏蔽 swap top1 test_r9r_new 0.688→0.738、test_r9q_new 0.746→0.775、test_r9s_new 0.773→0.818，carry 持平。
    MASK_ROUTE = False

    def __init__(self, deploy_dir: Path, run_dir: Optional[Path] = None, max_seeds: Optional[int] = None):
        import onnxruntime as ort
        sys.path.insert(0, str(PROJECT_DIR / "tools" / "cgnet"))
        import build_data as BD                        # noqa: E402  —— 与训练同一份特征构造
        self.BD = BD
        man = json.loads((deploy_dir / "manifest.json").read_text(encoding="utf-8"))
        self.manifest = man
        self.kinds = frozenset(man.get("kinds") or ["carry", "swap"])
        gpath = PROJECT_DIR / man["graph"]
        # **图谱必须是训练时那一版**（第 42 局：图谱子代理往 data/card_graph.json 加新边类型，关系通道 90→91+，
        # 每题 IndexError 退回 Laya）。部署包里放冻结副本，md5 对不上就拒绝加载。
        import hashlib
        md5 = hashlib.md5(gpath.read_bytes()).hexdigest()
        if man.get("graph_md5") and md5 != man["graph_md5"]:
            raise RuntimeError(f"cgnet 图谱版本不符：{gpath} md5={md5}，训练时 {man['graph_md5']}")
        BD.load_graph = lambda: json.loads(gpath.read_text(encoding="utf-8"))
        self.G = BD.Graph()
        self.G.path = str(gpath)
        self.builder = BD.Builder(self.G, None)
        # 训练时的"图谱外文本"表：同一段文本拿同一个下标；线上新出现的图谱外名字没有向量 → 文本下标 0（无文本）
        extra = json.loads((deploy_dir / "text_keys.json").read_text(encoding="utf-8"))
        self.builder.extra_texts = list(extra)
        self.builder.extra_idx = {t: i for i, t in enumerate(extra)}
        self.n_text = len(self.G.nodes) + 1 + len(extra)
        self.v2 = None
        if man.get("version") == "v2":
            import build_data_v2 as BD2                # noqa: E402
            self.v2 = BD2.V2(self.G)
        self.meta1 = {"tok_types": BD.TOK_TYPES, "n_channels": 2 * len(BD.EDGE_TYPES) + len(BD.EXTRA_CH)}
        self.n_ch = man["n_channels"]
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sessions = [ort.InferenceSession(str(deploy_dir / f), so, providers=["CPUExecutionProvider"])
                         for f in man["onnx"][:max_seeds or None]]
        self.talents = TalentTail(Path(run_dir) / "states.jsonl" if run_dir else None)

    def set_run_dir(self, run_dir: Path) -> None:
        self.talents = TalentTail(Path(run_dir) / "states.jsonl")

    # ------------------------------------------------------------ 特征
    def features(self, p: dict, owned_talents: Optional[list] = None) -> dict:
        """一道题 → 模型输入（numpy dict）。owned_talents=None 时从 run_dir/states.jsonl 增量回放。"""
        if owned_talents is None:
            owned_talents = self.talents.update()
        lab = {"dist": {k: 1.0 for k in p["options"]}, "best": p["options"][0]}   # 占位：线上没有老师标签
        # 线上题目没有数据集才有的 id / split（第 42 局 10 道题全部 KeyError('id') 退回 Laya）
        p = dict(p, id=p.get("id") or "live", split=p.get("split") or "live")
        if self.MASK_ROUTE and (p.get("ctx") or {}).get("route"):
            p = dict(p, ctx=dict(p["ctx"], route="", route_since=0))
        s = self.builder.convert(p, lab, owned_talents, None)
        if self.v2 is not None:
            import build_data_v2 as BD2                # noqa: E402
            old, BD2.MIN_ENGINE_FEAT = BD2.MIN_ENGINE_FEAT, bool(self.manifest.get("min_engine_feat"))   # 各部署包按训练时的开关
            old_e = getattr(BD2, "ECO_CORE_FEAT", False)
            BD2.ECO_CORE_FEAT = bool(self.manifest.get("eco_core_feat"))     # 2026-10-13 cs_r9zc：核心缺件各级命中特征
            old_f = getattr(BD2, "FLOW_FEAT", False)
            BD2.FLOW_FEAT = bool(self.manifest.get("flow_feat"))             # 2026-10-14：流派离成型还差什么（glob 103~108）
            old_g, old_nd = getattr(BD2, "ENGINE_FEAT", False), BD2.NUM_DIM2
            old_w = getattr(BD2, "W_FEAT", False)
            if self.manifest.get("w_feat"):                                    # 2026-10-16 f9lw：过渡件 / 回本 / 别家主C / 临时主C 宿主 / 等位缺件（num 44~48）
                _wv = int(self.manifest.get("w_feat"))                         # 2 = 再加场上杂牌（num 49）
                BD2.W_FEAT, BD2.NUM_DIM2 = (_wv if _wv > 1 else True), {2: 50, 3: 51, 4: 53}.get(_wv, 49)
            if self.manifest.get("engine_feat"):                               # 2026-10-14 cs_r9zd：推演器给每个选项的打分（num 44~47）
                BD2.ENGINE_FEAT, BD2.NUM_DIM2 = True, 48
                if "_engine" not in p:
                    p = dict(p, _engine=engine_annot(p))
            try:
                s = self.v2.augment(s, p, self.meta1)
            finally:
                BD2.MIN_ENGINE_FEAT = old
                BD2.ECO_CORE_FEAT = old_e
                BD2.FLOW_FEAT = old_f
                BD2.ENGINE_FEAT, BD2.NUM_DIM2 = old_g, old_nd
                BD2.W_FEAT = old_w
        s["text"] = np.where(s["text"] < self.n_text, s["text"], 0)
        return apply_ablation(s, self.manifest)

    def probs(self, p: dict, owned_talents: Optional[list] = None) -> dict:
        s = self.features(p, owned_talents)
        x = collate_np(s, self.n_ch)
        ps = []
        for sess in self.sessions:
            z = sess.run(None, x)[0][0].astype(np.float64)
            e = np.exp(z - z.max())
            ps.append(e / e.sum())
        pr = np.mean(ps, axis=0)
        return {k: float(v) for k, v in zip(s["keys"], pr)}

    def ask(self, p: dict) -> Optional[tuple]:
        """与 LayaPolicy._ask 同格式：(choice, probs, conf, ms)。出错返回 None（调用方退回 Laya）。"""
        try:
            t0 = time.perf_counter()
            pr = self.probs(p)
            choice = max(pr, key=pr.get)
            return choice, pr, float(pr[choice]), round((time.perf_counter() - t0) * 1000, 1)
        except Exception:                                          # noqa: BLE001
            logger.exception("cgnet 推理失败（%s）—— 这道题退回 Laya", p.get("kind"))
            return None


class CGNetPolicy:
    """按题型路由到各自的部署包：data/cgnet/deploy/router.json = {"carry": "carry", "swap": "swap"}（题型 -> 子目录）。
    没有 router.json 时把 deploy_dir 本身当成一个部署包（manifest.json 里的 kinds 全接）。"""

    def __init__(self, models: dict):
        self.models = models                                   # 题型 -> CGNetModel
        self.kinds = frozenset(models)

    @classmethod
    def load(cls, deploy_dir: Optional[Path] = None, run_dir: Optional[Path] = None,
             max_seeds: Optional[int] = None) -> Optional["CGNetPolicy"]:
        """进程启动时调用一次。任何失败（没导出 / onnxruntime 缺失 / 数据表对不上）→ None，调用方退回 Laya。
        max_seeds：每个部署包只用前几个种子（默认全部 3 个取概率平均；1 个更快）。"""
        try:
            t0 = time.time()
            d = Path(deploy_dir or DEPLOY_DIR)
            router = d / "router.json"
            models: dict = {}
            if router.exists():
                cache: dict = {}
                for kind, sub in json.loads(router.read_text(encoding="utf-8")).items():
                    if sub not in cache:
                        cache[sub] = CGNetModel(d / sub, run_dir, max_seeds)
                    models[kind] = cache[sub]
            else:
                m = CGNetModel(d, run_dir, max_seeds)
                models = {k: m for k in m.kinds}
            pol = cls(models)
            logger.info("cgnet 已加载 %.1fs：%s", time.time() - t0,
                        {k: f"{m.manifest.get('run')}({m.manifest.get('version')}, {len(m.sessions)} 种子)" for k, m in models.items()})
            return pol
        except Exception:                                          # noqa: BLE001
            logger.exception("cgnet 加载失败 —— carry/swap 退回 Laya")
            return None

    def set_run_dir(self, run_dir: Path) -> None:
        for m in set(self.models.values()):
            m.set_run_dir(run_dir)

    def probs(self, p: dict, owned_talents: Optional[list] = None) -> dict:
        return self.models[p["kind"]].probs(p, owned_talents)

    def ask(self, p: dict) -> Optional[tuple]:
        """与 LayaPolicy._ask 同格式：(choice, probs, conf, ms)。不接管的题型 / 出错 → None（调用方退回 Laya）。"""
        m = self.models.get(p.get("kind"))
        return m.ask(p) if m is not None else None
