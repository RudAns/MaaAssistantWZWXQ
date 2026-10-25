"""手牌卡名读取：服务器版认字模型 + 闭集 CTC 解码（输出只可能是卡库里的真卡名）。

2026-09-25：原始轻量 OCR 准确度偏低；结合图像识别看哪里是卡名、有没有被截断，用更好的模型或算法减少误判。
实例：N 选 1 刚拿到的亚瑟，手牌没叠放、卡名清清楚楚，轻量 OCR 自由解码读成「业热」→ 认不出、整回合没上场。

测试集难例复核（tools/ocr/eval_hand_names.py，名字条上能定真值的 60 条）：
  轻量自由解码 + 模糊匹配（原线上）：对 25 / 读不出 28 / **读错 7**（雷鸣刃→驱雷 ×3、力量腰带→新生力量 ×3、干将莫邪→牛魔）
  轻量 + 闭集：22 / 37 / 1；**服务器版 + 闭集：28 / 32 / 0**。
叠放（≥6 张）时卡名在牌面正中、基本被下一张盖住 —— 那些条子**任何读字方法都定不了**，交给点开看 / 立绘识别。

口径（与 eval_hand_names.decide 一致）：
- 露出宽度 ≥ FULL_W：只在完整卡名里选；更窄：连前缀一起选（前缀当作"图上恰好就这几个字"）。
- 第一名比第二名高不到 MARGIN（log 概率）→ 读不出（宁缺勿错）。
- 前缀只对应一张牌 → 就是它；对应多张 → 返回 ("", 前缀)，调用方再用"最近见过/商店里的牌"去唯一化。
候选先按"首字在这张图上出现过的概率"剪枝（整表 800+ 卡名逐个算 CTC 要 ~200ms/条，剪完只剩几个到几十个）；结果按像素缓存。
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = ROOT / "assets" / "model" / "ocr_server"
FULL_W = 110
MARGIN = 2.0
EMPTY_MIN = 0.0                 # 卡名得分至少不低于"全空白"路径（见 read()）
FIRST_CHAR_MIN = 1e-3
REC_H = 48
logger = logging.getLogger(__name__)

sys.path.insert(0, str(ROOT / "tools" / "ocr"))
from ctc_closed import ctc_logp  # noqa: E402


class HandNameReader:
    _inst: Optional["HandNameReader"] = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> Optional["HandNameReader"]:
        with cls._lock:
            if cls._inst is None:
                try:
                    cls._inst = cls()
                except Exception:                           # noqa: BLE001
                    logger.exception("手牌卡名读取器（服务器版认字 + 闭集解码）加载失败，退回原读法")
                    cls._inst = False                       # type: ignore[assignment]
            return cls._inst or None

    def __init__(self) -> None:
        from . import onnx_ocr
        base = onnx_ocr.OnnxOcr.get()
        self.rec = onnx_ocr._Runner("rec", base.ep, model_dir=SERVER_DIR)
        keys = (SERVER_DIR / "keys.txt").read_text(encoding="utf-8").split("\n")
        if keys and keys[-1] == "":
            keys = keys[:-1]
        chars = ["<blank>"] + keys + [" "]
        idx = {c: i for i, c in enumerate(chars)}
        cards = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))["cards"]
        # 召唤物不会出现在手牌里（它们和同名英雄共用前缀，会把"马可波"这类前缀弄成"歧义"）
        names = [n for n, c in cards.items() if c.get("category") in ("英雄牌", "效果牌", "装备牌", "天赋牌", "棋手技能牌")]
        self.cands: list[tuple[str, list[int], bool]] = []        # (串, 字下标, 是否完整卡名)
        seen = set()
        for n in names:
            ids = [idx.get(ch) for ch in n]
            if not n or any(i is None for i in ids):
                continue
            for k in range(1, len(ids) + 1):
                s = n[:k]
                if s in seen and k < len(ids):
                    continue
                seen.add(s)
                self.cands.append((s, ids[:k], k == len(ids)))
        pref: dict = {}
        for n in names:
            for k in range(1, len(n)):
                pref.setdefault(n[:k], set()).add(n)
        self.unique_prefix = {p: next(iter(s)) for p, s in pref.items() if len(s) == 1}
        self._cache: dict = {}
        logger.info("手牌卡名读取器：服务器版认字（%s）+ 闭集解码，候选 %d 个", base.ep, len(self.cands))

    def _probs(self, crop_bgr: np.ndarray) -> np.ndarray:
        import cv2
        h, w = crop_bgr.shape[:2]
        ww = max(8, int(round(w * REC_H / max(1, h))))
        W = (max(64, min(960, ww)) + 7) // 8 * 8
        x = np.zeros((1, 3, REC_H, W), np.float32)
        im = cv2.resize(crop_bgr, (min(ww, W), REC_H)).astype(np.float32) / 255.0
        x[0, :, :, :min(ww, W)] = ((im - 0.5) / 0.5).transpose(2, 0, 1)
        return self.rec(x)[0]

    def read(self, crop_bgr: np.ndarray, width: int) -> tuple[str, str, float]:
        """→ (卡名, 前缀, 余量)。卡名非空 = 认定；卡名空、前缀非空 = 只认出多张牌共用的前缀；都空 = 读不出。"""
        key = hashlib.md5(crop_bgr.tobytes()).hexdigest() + f":{width}"
        if key in self._cache:
            return self._cache[key]
        probs = self._probs(crop_bgr)
        lp = np.log(np.clip(probs, 1e-12, 1.0))
        pmax = probs.max(0)
        trunc = width < FULL_W
        scored = []
        for s, ids, full in self.cands:
            if pmax[ids[0]] < FIRST_CHAR_MIN or (not trunc and not full):
                continue
            scored.append((ctc_logp(lp, ids), s, full))
        scored.sort(reverse=True)
        res = ("", "", 0.0)
        if scored:
            # 每个候选指向哪张牌：完整卡名 / 唯一前缀 → 那张牌；多张共用的前缀 → 这个前缀本身。
            # 余量**只和"指向别的牌"的最佳候选比**：截断时「速击之」「速击」都指向速击之枪，
            # 末字被切掉一半时两者分数接近 —— 它们不是竞争对手（原来因此把 21 条能认的判成读不出）。
            tgt = lambda s, full: s if full else self.unique_prefix.get(s, "?" + s)   # noqa: E731
            best, s0, f0 = scored[0]
            t0 = tgt(s0, f0)
            rival = next((sc for sc, s, f in scored[1:] if tgt(s, f) != t0), -1e9)
            # 「什么字都没有」也是一个对手（2026-10-01 第 50~52 局：布局多推断出一张牌，空背景上剪枝后只剩「海月」一个候选，
            # 第二名是 -1e9、余量等于无穷 → 三局开局都读出一张不存在的海月并当成主C）。空串路径 = 全程空白的概率。
            # 单设门槛 EMPTY_MIN=0（测试集 346 条：只比第二名 对138/错4/空白读出1；加"不低于空串" 对136/错0/空白0；
            # 把空串当普通对手、同样要高 2.0 则对124，代价太大）。线上那张幻影「海月」比空串低 10.4。
            empty = ctc_logp(lp, [])
            margin = best - rival
            if margin >= MARGIN and best - empty >= EMPTY_MIN:
                res = (t0, "", float(margin)) if not t0.startswith("?") else ("", t0[1:], float(margin))
        if len(self._cache) > 4000:
            self._cache.clear()
        self._cache[key] = res
        return res


# v2（2026-10-12）：真实样本覆盖 420 局 + 官网卡面合成，词表 84 → 551（芈月等）；v1 文件保留，回滚改回 hand_art_v1.onnx。
ART_MODEL = ROOT / "assets" / "model" / "vision" / "hand_art_v2.onnx"


class HandArtReader:
    """手牌**立绘**识别（tools/vision/train_hand_art.py）：叠放时卡名被盖住，靠露出来的左边一截（图标 + 立绘）认牌。
    训练标签 = 不叠放时读得准的卡名（左边一截叠不叠放都一样）。测试集（不叠放）：词表内 100%，门槛下被采纳 71%、准确 98.9%；
    叠放牌上与名字条露出的前缀对照：一致 753 / 矛盾 9；另有 476 张名字条一个字都看不到 —— 以前只能点开看。
    v2：同一测试集 被采纳 71% → 99.6%、准确 98.9% → 100%；最近 20 局 采纳 53% → 98.4%、准确 100%（门槛 0.7）。
    输入与训练一致：RGB 帧 [595:715, 图标x-12 : 图标x+52] → 中心 56x112。模型不在返回 None。"""
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            try:
                import onnxruntime as ort
                meta = json.loads(ART_MODEL.with_suffix(".json").read_text(encoding="utf-8"))
                cls._inst = cls()
                cls._inst.sess = _gpu_session(ART_MODEL, "手牌立绘")
                cls._inst.classes, cls._inst.thresh = meta["classes"], float(meta["thresh"])
            except Exception:                               # noqa: BLE001
                logger.exception("手牌立绘识别加载失败（只用读字）")
                cls._inst = False
        return cls._inst or None

    def read(self, image_rgb: np.ndarray, icon_x: float) -> tuple[str, float]:
        x0 = int(icon_x) - 12
        if x0 < 0 or x0 + 64 > image_rgb.shape[1]:
            return "", 0.0
        c = image_rgb[595 + 4:595 + 116, x0 + 4:x0 + 60]
        x = np.ascontiguousarray(c.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)
        lg = self.sess.run(None, {"x": x})[0][0]
        e = np.exp(lg - lg.max())
        p = e / e.sum()
        k = int(p.argmax())
        return (self.classes[k], float(p[k])) if p[k] >= self.thresh else ("", float(p[k]))

    def read_top(self, image_rgb: np.ndarray, icon_x: float) -> tuple[str, float]:
        """不设门槛的第一选择（给"账本里该有的那几张"做闭集核对用：extractors._hand_name_closed）。"""
        x0 = int(icon_x) - 12
        if x0 < 0 or x0 + 64 > image_rgb.shape[1]:
            return "", 0.0
        c = image_rgb[595 + 4:595 + 116, x0 + 4:x0 + 60]
        x = np.ascontiguousarray(c.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)
        lg = self.sess.run(None, {"x": x})[0][0]
        e = np.exp(lg - lg.max())
        p = e / e.sum()
        k = int(p.argmax())
        return self.classes[k], float(p[k])



def _gpu_session(path, name):
    from .gpu_session import session
    return session(path, name)

def _cpu_so():
    """CPU 上的小模型一律限 2 线程（2026-10-12：没设时 onnxruntime 默认开满 24 核，手牌 / 场上一多就把整机冲到 70~90%）。"""
    import onnxruntime as _ort
    so = _ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    return so
