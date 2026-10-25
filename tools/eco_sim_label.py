"""经济节奏题（eco：levelup / roll / save）的**模拟标签**（2026-10-10 晚：用已有的数学模型模拟出标签）。

不再让老师凭经验标，直接用商店概率模拟算：从每道题当时的真实局面（回合、能量、棋手等级 / 经验、升级费、免费刷新、
已有的人、缺的人和品阶、主C 在不在）出发，三个选项各模拟 N 次：
  - 本回合按选项做（levelup：先升一级，再看一眼商店买缺的人；roll：免费刷新 + 付费刷新直到刷完买不起英雄，见到缺的人就买；
    save：什么都不花）；
  - 之后每回合按模拟最优的 S2 做法（tools/sim_route_vs_flex.py）：落后进度表且升得起就升级，然后有钱就刷、见到缺的人就买；
  - 到 min(本回合 + HORIZON, 12) 回合看结果：
      score = 拿到的缺的人（主C 记 2，配套记 1）+ 0.5 × 人口上限增量 + 0.15 × 棋手等级增量 + 0.03 × 剩余能量。
三个选项按平均分做 softmax（温度 TAU）得到软标签，格式与老师标签相同（id / dist / best / confidence / reason）。

常量全部来自 agent/decision/meta.py（实测商店概率、牌库、收入、经验）。对手每回合从牌库拿走 OPP_TAKE 张。
用法：
  .venv/Scripts/python.exe -X utf8 tools/eco_sim_label.py --ds r9t2 --out data/laya/labels/r9t_sim/labels.jsonl [--n 200] [--workers 8]
"""
import argparse
import json
import math
import multiprocessing as mp
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.decision import meta  # noqa: E402

CARDS = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))["cards"]
HEROES = {n: v["quality"] for n, v in CARDS.items() if v.get("category") == "英雄牌" and v.get("quality")}
FACTION = {n: (v.get("faction") or "") for n, v in CARDS.items()}
OPP_TAKE = 14
# 商店里的"发英雄牌"（2026-10-10：有些效果牌给特定标签的英雄，有些随机，有些可选）：
# 卡池来自图谱 v3 的 grants_pool 边（random n / choose:s选n），出现频率按最近 40 局 8485 个商店格子实测（每格约 1.1%）
_G = json.loads((ROOT / "data" / "card_graph_v3.json").read_text(encoding="utf-8"))
GRANTS = {}
for _e in _G.get("edges") or []:
    if _e.get("type") == "grants_pool":
        _a = _e.get("attrs") or {}
        GRANTS.setdefault(_e["from"], {"pick": str(_a.get("pick") or "random"), "n": int(_a.get("n") or 1), "pool": []})["pool"].append(_e["to"])
GRANT_SHOP = {"图腾仪式": 39, "广纳良才": 34, "乱世": 16, "牺牲品": 1}      # 实测出现次数（权重）
GRANT_SLOT_P = 0.011
HORIZON = 4
TAU = 0.35
MERGE_CARRY = 0.6                 # 主C 买到同名（合成升星）的价值
MERGE_OTHER = 0.25                # 其他已有英雄买到同名
MERGE_MAX = 4                     # 每人最多再算几张同名
BAN_NORM = {"sanfen": "三分之地", "heluo": "河洛", "dahe": "大河流域", "rilu": "日落海", "zhulu": "逐鹿"}
POP_CAP = {1: 5, 2: 5, 3: 6, 4: 6, 5: 7, 6: 7}
XP_NEED = {1: 7, 2: 13, 3: 17, 4: 23, 5: 27}         # 本级升下一级所需经验（经验条 N/M 的 M）


def draw(pool, level, rng):
    odds = meta.TIER_ODDS.get(min(level, 6)) or {1: 1.0}
    tiers, w = zip(*odds.items())
    t = rng.choices(tiers, w)[0]
    names = [n for n, q in HEROES.items() if q == t and pool[n] > 0]
    if not names:
        return None
    return rng.choices(names, [pool[n] for n in names])[0]


def shop(pool, level, rng):
    out = []
    for _ in range(meta.SHOP_SLOTS):
        if rng.random() < GRANT_SLOT_P:
            out.append("#" + rng.choices(list(GRANT_SHOP), list(GRANT_SHOP.values()))[0])
        else:
            out.append(draw(pool, level, rng))
    return out


class Sim:
    def __init__(self, ctx, missing, owned, carry_missing, rng, carry="", family=(), board_n=0, waiting_n=0):
        self.rng = rng
        self.r = int(ctx.get("round") or 1)
        self.gold = int(ctx.get("gold") or 0)
        self.lv = int(ctx.get("level") or 1)
        self.xp = int(ctx.get("xp") or 0)
        self.need = int(ctx.get("xp_needed") or XP_NEED.get(self.lv, 27))
        lc = ctx.get("level_cost")
        self.cost_now = int(lc) if isinstance(lc, int) and lc > 0 else max(1, self.need - self.xp)
        self.free = int(ctx.get("free_refresh") or 0)
        ban = BAN_NORM.get(str(ctx.get("banned") or ""), str(ctx.get("banned") or ""))
        self.pool = {n: meta.POOL_COPIES[q] for n, q in HEROES.items()}
        for n in self.pool:
            if ban and FACTION.get(n) == ban:
                self.pool[n] = 0
        for n in owned:
            if n in self.pool:
                self.pool[n] = max(0, self.pool[n] - 1)
        self.missing = {m[0]: (2.0 if (m[0] == carry_missing) else 1.0) for m in missing if m[0] in HEROES}
        # **合成升星也是收益**（v2，2026-10-11：第 535 局后期阵容齐了、模拟里刷牌没收益，eco 学成"后期攒钱"，34 能量不花）：
        # 已有的体系英雄买到同名 = 合成升级；主C 同名 MERGE_CARRY、其他 MERGE_OTHER（每人最多再要 MERGE_MAX 张）
        # 只算**主C 和路线最终阵容里的人**的同名：过渡件（弈星 / 苏烈 / 1 阶临时主C）合成了后面也要卖，不算收益
        # 1 阶临时主C（不在路线阵容里）会被路线主C 换掉：它的同名不算收益
        real_carry = carry and (carry in set(family or ()) or HEROES.get(carry, 1) >= 2)
        self.family = set(family or ()) | ({carry} if real_carry else set())
        self.merge = {n: MERGE_MAX for n in owned if n in HEROES and n in self.family}
        self.carry = carry
        self.got = 0.0
        # **人口**（v3，2026-10-11：手牌已拿到核心英雄、场上核心只差一个就凑满时，应优先攒钱升级）：
        # 体系英雄"拿到"记一半分，"上场"再记一半；人口满时进备战区，升级扩人口后上场才补另一半。
        # 开局手里就有、场上没位置的体系英雄算在备战区（只差人口的核心）。
        self.board = int(board_n or 0)
        self.bench = [1.0 for _ in range(int(waiting_n or 0))]
        self.lv0 = self.lv

    def cap(self):
        return POP_CAP.get(min(self.lv, 6), 7)

    def place(self, w):
        if self.board < self.cap():
            self.board += 1
            self.got += 0.5 * w
        else:
            self.bench.append(w)

    def field_bench(self):
        while self.bench and self.board < self.cap():
            self.board += 1
            self.got += 0.5 * self.bench.pop(0)

    def level_up(self):
        if self.lv >= 6 or self.gold < self.cost_now:
            return False
        self.gold -= self.cost_now
        self.lv += 1
        self.xp = 0
        self.need = XP_NEED.get(self.lv, 27)
        self.cost_now = self.need
        self.field_bench()
        return True

    def grant(self, card):
        """买一张发英雄牌：random = 从卡池（剩余张数加权）随机发 n 张；choose:s选n = 抽 s 张、挑缺的人最多 n 张。"""
        g = GRANTS.get(card)
        if not g:
            return
        cands = [h for h in g["pool"] if self.pool.get(h, 0) > 0]
        if not cands:
            return
        import re as _re
        m = _re.match(r"choose:(\d+)选(\d+)", g["pick"])
        if m:
            s_, take = int(m.group(1)), int(m.group(2))
            shown = self.rng.sample(cands, min(s_, len(cands)))
            got = [h for h in shown if h in self.missing][:take]
        else:
            got = self.rng.choices(cands, [self.pool[h] for h in cands], k=g["n"])
        for h in got:
            if h in self.missing:
                self.got += self.missing.pop(h)
                self.pool[h] = max(0, self.pool[h] - 1)

    def look(self, items):
        for h in items:
            if h and h.startswith("#"):
                card = h[1:]
                cost = int((CARDS.get(card) or {}).get("cost") or 2)
                g = GRANTS.get(card) or {}
                if self.gold >= cost and any(x in self.missing for x in g.get("pool") or []):
                    self.gold -= cost
                    self.grant(card)
                continue
            if h and h in self.missing and self.gold >= 3:
                self.gold -= 3
                w = self.missing.pop(h)
                self.got += 0.5 * w
                self.place(w)
                self.pool[h] = max(0, self.pool[h] - 1)
                if h in self.family:
                    self.merge[h] = MERGE_MAX
            elif h and self.merge.get(h, 0) > 0 and self.gold >= 3:
                self.gold -= 3
                self.merge[h] -= 1
                self.got += MERGE_CARRY if h == self.carry else MERGE_OTHER
                self.pool[h] = max(0, self.pool[h] - 1)

    def roll(self, reserve=3):
        self.look(shop(self.pool, self.lv, self.rng))
        while self.missing or any(v > 0 for v in self.merge.values()):
            if self.free > 0:
                self.free -= 1
            elif self.gold - 2 >= reserve:
                self.gold -= 2
            else:
                break
            self.look(shop(self.pool, self.lv, self.rng))

    def end_round(self):
        # 回合结束：经验 +1（拍卖前 +3）；对手拿牌；下回合收入、1 次免费刷新
        self.xp += meta.lord_xp_gain(self.r)
        while self.lv < 6 and self.xp >= self.need:
            self.xp -= self.need
            self.lv += 1
            self.need = XP_NEED.get(self.lv, 27)
        self.field_bench()
        self.cost_now = max(1, self.need - self.xp)
        for _ in range(OPP_TAKE):
            h = draw(self.pool, self.lv, self.rng)
            if h:
                self.pool[h] -= 1
        self.r += 1
        self.gold += meta.energy_income(self.r)
        self.free = 1

    def default_round(self):
        if self.lv < meta.level_target(self.r) and self.gold >= self.cost_now:
            self.level_up()
        self.roll()

    def run(self, first):
        if first == "levelup":
            if not self.level_up():
                pass                                          # 升不起 = 什么都不做（同 save）
            else:
                self.look(shop(self.pool, self.lv, self.rng))
        elif first == "roll":
            self.roll()
        end = self.r + HORIZON                                  # v2：不再卡在第 12 回合（后期题原来推演 0 回合）
        self.end_round()
        while self.r <= end:
            self.default_round()
            if self.r == end:
                break
            self.end_round()
        dpop = POP_CAP.get(min(self.lv, 6), 7) - POP_CAP.get(min(self.lv0, 6), 5)
        return self.got + 0.2 * dpop + 0.15 * (self.lv - self.lv0) + 0.03 * self.gold


def label_one(args):
    p, n, seed = args
    ctx = p.get("ctx") or {}
    raw = p.get("raw") or {}
    missing = raw.get("missing") or []
    carry = raw.get("carry") or ""
    owned = [u for u in (p.get("field") or [])] + [h for h in (ctx.get("hand") or []) if h in HEROES]
    carry_missing = next((m[0] for m in missing if "CARRY" in str(m[2]).upper() or "主C" in str(m[2])), "") if not carry else ""
    opts = [o for o in (p.get("options") or ["levelup", "roll", "save"])]
    from agent.decision import routes as _R
    family = set(_R.final_team(_R.by_key(ctx.get("route") or "") or {}))
    field_heroes = [u for u in (p.get("field") or []) if u in HEROES]
    board_n = len(field_heroes)
    fam_c = family | ({carry} if carry else set())
    waiting_n = len([h for h in (ctx.get("hand") or []) if h in HEROES and h in fam_c and h not in field_heroes])
    means = {}
    for i, o in enumerate(opts):
        rng = random.Random(seed * 1000 + i)
        xs = [Sim(ctx, missing, owned, carry_missing, rng, carry, family, board_n, waiting_n).run(o) for _ in range(n)]
        means[o] = st.mean(xs)
    m = max(means.values())
    ex = {o: math.exp((v - m) / TAU) for o, v in means.items()}
    z = sum(ex.values())
    dist = {o: round(v / z, 3) for o, v in ex.items()}
    best = max(dist, key=dist.get)
    reason = "模拟 v3（%d 次/选项，看到第 %d 回合，含合成升星 / 人口上场）平均分 " % (n, int(ctx.get("round") or 1) + HORIZON) + \
             "，".join(f"{o} {v:.2f}" for o, v in means.items())
    return {"id": p["id"], "dist": dist, "best": best, "confidence": dist[best], "reason": reason, "means": means}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="r9t2")
    ap.add_argument("--out", default="data/laya/labels/r9t_sim/labels.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    pts = [json.loads(l) for l in open(ROOT / "data" / "laya" / "datasets" / a.ds / "points.jsonl", encoding="utf-8")]
    pts = [p for p in pts if p.get("kind") == "eco"]
    if a.limit:
        pts = pts[:a.limit]
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(p, a.n, i + 1) for i, p in enumerate(pts)]
    with mp.Pool(a.workers) as pool, open(out, "w", encoding="utf-8") as f:
        for k, r in enumerate(pool.imap(label_one, jobs, chunksize=4)):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if (k + 1) % 200 == 0:
                print(f"{k + 1}/{len(jobs)}", flush=True)
    labs = [json.loads(l) for l in open(out, encoding="utf-8")]
    from collections import Counter
    print("题数", len(labs), "最佳分布", Counter(l["best"] for l in labs))


if __name__ == "__main__":
    main()
