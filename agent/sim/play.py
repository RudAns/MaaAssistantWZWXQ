"""推演里"以后每回合怎么打"（默认打法）。像人一样灵活：主C 不定死，谁合适用谁；撞运气的牌往后推几步看命中再选。

唯一的判断输入是"谁适合当主C"的强度表（data/hero_carry_reviewed.json 的 carry_score，来自卡面分析 + 人工经验，可改），
其余全是养成账（等级、觉醒、装备、钱）。

  from agent.sim.play import Player, simulate
  runs = simulate(state, n=100, rounds_to=14)
"""
from __future__ import annotations

import json
import math
import re
import random
from functools import lru_cache
from pathlib import Path

from agent.engine import board as B
from .engine import Engine, card_db, category, quality, impl, card_text
from .state import State

ROOT = Path(__file__).resolve().parents[2]
HOLD = {"遗志", "传承"}           # 搬等级的牌：默认先留着，换主C 时才用
_REMOVE_RE = re.compile(r"移除|变为|变成")    # 卡面会让目标离场 / 变成别人的牌


@lru_cache(maxsize=1)
def carry_rating() -> dict:
    try:
        r = json.loads((ROOT / "data/hero_carry_reviewed.json").read_text(encoding="utf-8"))
        r = r.get("heroes", r)
        return {k: float(v.get("carry_score", 0)) for k, v in r.items() if isinstance(v, dict)}
    except Exception:  # noqa: BLE001
        return {}


@lru_cache(maxsize=1)
def event_offers() -> dict:
    try:
        return json.loads((ROOT / "data/engine/event_offers.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


@lru_cache(maxsize=1)
def hero_roles() -> dict:
    try:
        d = json.loads((ROOT / "data/hero_roles.json").read_text(encoding="utf-8"))
        return d.get("heroes", d)
    except Exception:  # noqa: BLE001
        return {}


OUTPUT_ROLES = {"carry_phys", "carry_mag", "assassin", "bruiser"}


def carry_ok(name: str) -> bool:
    """能不能当主C（推演打法用）。2026-10-14：技能强力的单核只要能养起来，在阿离套路里都很强（玄策也可以），
    一般 3 阶以上的输出英雄只要等级起来都很强 → 输出位（data/hero_roles.json 按卡面分析的 combat_role）且 2 阶以上；
    1 阶是过渡件。强弱主要看养到几级，评分只是小权重（见 Player.pick_carry / dev_score）。"""
    from .engine import quality
    n = name.replace("(赠送)", "")
    roles = set((hero_roles().get(n) or {}).get("combat_role") or [])
    return bool(roles & OUTPUT_ROLES) and (quality(n) or 0) >= 2


def rating(name: str) -> float:
    return carry_rating().get(name.replace("(赠送)", ""), 0.0)


def _by_item_fit(item: str, targets: list, carry: str) -> list:
    """装备给谁：按装备 × 英雄适配表（hero_traits.item_fit_score，data/equip_relations.json）排，适配为负的不给
    （机器头强制攻速 0.4 这类负面效果给主C 是亏的，2026-10-14）。主C 适配 ≥5 时主C 优先，否则谁最适配给谁。"""
    try:
        from agent.decision.hero_traits import item_fit_score
    except Exception:  # noqa: BLE001
        return targets
    sc = {id(t): item_fit_score(item, t.name) for t in targets}
    keep = [t for t in targets if sc[id(t)] is None or sc[id(t)] >= 0]
    def key(t):
        v = sc[id(t)]
        v = 4.0 if v is None else v
        return (-(1 if (t.name == carry and v >= 5) else 0), -v)
    return sorted(keep, key=key)


def dev_score(st: State) -> float:
    """推演打法自己用的"养成好坏"：主C 的强度评分 × log(1+等级)（觉醒 ×1.5）+ 全队等级的小权重。
    只用来在推演里做选择（打法），不是对外的价值判断。"""
    c = st.unit(st.carry) if st.carry else None
    v = 0.0
    if c:
        # 阿离是单C：资源集中到一个人身上有回报 → 主C 等级用 0.8 次方（原来取对数，众星拱月"全队各 10 级给主C"永远不划算，
        # 推演 100 局拍卖里出现 7 次只拍了 1 次；实战第 12 回合用它的局约 18%）
        w = 1.0 if carry_ok(c.name) else 0.4                 # 合格的输出位之间不再按评分拉开（养到几级才是关键）
        v += 2.0 * w * (1.0 + c.level) ** 0.8 * (1.5 if c.awakened else 1.0)
    v += 0.3 * sum(math.sqrt(max(0.0, u.level)) for u in st.heroes() if u is not c)
    v += 0.02 * st.energy
    return v


class Player:
    def __init__(self, eng: Engine, events: bool = True, lookahead: int = 2, samples: int = 2):
        self.e = eng
        self.events = events
        self.lookahead = lookahead
        self.samples = samples
        self.no_swap_round = None      # 标注换人题用：这一回合的"卖谁腾位"已经由题目选项定了，打法不再自己换人
        eng.chooser = self.choose
        if not eng.s.lord_cards:
            eng.s.lord_cards = ["枫叶舞"]
        if eng.s.lord == "阿离" and "惊鸿游龙" not in eng.s.lord_cards:
            eng.s.lord_cards.append("惊鸿游龙")         # 阿离秘技：开局自带（真实对局每局平均拿 2~3 张翩跹舞步）
        if eng.s.round == 1:
            eng.game_start()                             # 开局手里一件随机基础装备

    # ------------------------------------------------------------ 选择（撞运气的牌：往后推几步看）
    # 影响整局的选择（天赋、翩跹舞步、枫叶舞强化 / 特性、拍卖）：收益是一回合回合累积的（如翩跹舞步·蓄势），
    # 只往后看 1 回合会选得近乎随机 → 看更远、多抽几次
    LONG_KINDS = {"talent", "翩跹舞步", "lord_boost", "maple_trait", "auction"}
    LONG_LOOKAHEAD, LONG_SAMPLES = 4, 3

    def choose(self, eng, kind, options, **ctx):
        if len(options) <= 1 or self.lookahead <= 0:
            return options[0] if options else None
        la, ns = self.lookahead, self.samples
        if kind in self.LONG_KINDS:
            la, ns = max(la, self.LONG_LOOKAHEAD), max(ns, self.LONG_SAMPLES)
        seeds = [eng.rng.random() for _ in range(ns)]      # 所有选项共用同一批随机数：只比较选项本身的差别
        best, bv = options[0], -1e9
        vals = {}
        cand = list(options) + ([None] if kind == "talent" else [])   # 天赋多算一个"什么都不拿"，用来判断要不要刷新
        for o in cand:
            tot = 0.0
            for sd in seeds:
                s2 = eng.s.clone(keep_log=False)
                e2 = Engine(s2, random.Random(sd))
                p2 = self._sub(e2)
                try:
                    if o is not None:
                        p2._take(kind, o)
                    for _ in range(la):
                        p2.round()
                except Exception:  # noqa: BLE001
                    continue
                tot += dev_score(s2)
            vals[o] = tot / max(1, len(seeds))
            if o is not None and tot > bv:
                best, bv = o, tot
        self._last_vals = vals
        return best

    def cell_for(self, name: str):
        """上场落点：主C 后排中间；"距离最近的英雄"类贴着主C；前排 / 坦克标签放前排；其余中间两排。
        站位影响阵亡率（前排死得多）、"最近"类效果的对象、夺取、图腾光环。"""
        e, s = self.e, self.e.s
        free = e.free_cells()
        if not free:
            return None
        rows = lambda c: B.row_of(c)  # noqa: E731
        mid = lambda c: abs(B.BOARD_ROWS[0].index(c) - 3) if c in B.BOARD_ROWS[0] else abs(next(r.index(c) for r in B.BOARD_ROWS if c in r) - 3)  # noqa: E731
        carry = s.unit(s.carry) if s.carry else None
        tags = set((card_db().get(name) or {}).get("tags") or [])
        if name == s.carry or (carry is None and rating(name) >= 6):
            return min(free, key=lambda c: (-rows(c) if rows(c) <= 2 else -2, mid(c)))          # 后排（第 3 行优先，再第 4 行）
        if carry is not None and carry.cell and "距离最近" in card_text(name):
            return min(free, key=lambda c: (B.hex_dist(carry.cell, c), mid(c)))
        if tags & {"前排", "坦克"}:
            return min(free, key=lambda c: (rows(c), mid(c)))
        return min(free, key=lambda c: (abs(rows(c) - 1.5), mid(c)))

    def arrange(self) -> None:
        """准备阶段结束前整理站位（实战可以随便挪）：主C 不在后两排就挪到后排空格；"距离最近的英雄"类挪到离主C 最近的空格。"""
        e, s = self.e, self.e.s
        c = s.unit(s.carry) if s.carry else None
        if c is None:
            return
        free = e.free_cells()
        if c.cell is None or B.row_of(c.cell) < 2:
            back = [x for x in free if B.row_of(x) >= 2]
            if back:
                c.cell = min(back, key=lambda x: (-B.row_of(x) if B.row_of(x) <= 2 else -2, abs(x[0] - 640)))
        for u in s.heroes():
            if u is c or "距离最近" not in card_text(u.name) or not c.cell:
                continue
            free = e.free_cells()
            if free and (u.cell is None or B.hex_dist(u.cell, c.cell) > 1):
                best = min(free, key=lambda x: B.hex_dist(c.cell, x))
                if u.cell is None or B.hex_dist(best, c.cell) < B.hex_dist(u.cell, c.cell):
                    u.cell = best

    def buy_card(self, name: str) -> bool:
        """商店里的非英雄牌（效果 / 天赋 / 装备）：默认打法只买会产出等级的（卡面写着"等级"），且买完还留一张英雄的钱。"""
        e, s = self.e, self.e.s
        if len(s.hand) >= B.HAND_LIMIT - 1 or "等级" not in card_text(name) or impl(name) is None:
            return False
        if s.energy - e.shop_cost(name) < B.HERO_COST:
            return False
        return e.buy(name)

    def protected(self) -> set:
        """变身 / 移除类牌不许打的人：主C、枫叶舞宿主（子类加上自己的，如流派的仓库和最终主C）。"""
        s = self.e.s
        return {s.carry, s.maple} - {""}

    def _sub(self, eng):
        """往后看用的打法：同一套打法、不再嵌套往后看、不触发随机事件（子类覆盖以带上自己的参数）。"""
        return Player(eng, events=False, lookahead=0)

    def _take(self, kind, name):
        e = self.e
        if kind == "talent":
            e.s.talents.append(name)
            c = impl(name)
            if c and hasattr(c, "on_use"):
                e.use_card(name, None)
        elif kind == "lord_boost":
            e.s.lord_cards.append(name)
        else:
            if category(name) == "英雄牌":
                e.take(name)
            e.gain_card(name)
        self.play_hand()

    # ------------------------------------------------------------ 主C
    def pick_carry(self) -> None:
        s = self.e.s
        cands = [u for u in s.heroes() if carry_ok(u.name)]
        if not cands:
            cands = list(s.heroes())
        if not cands:
            return
        # 合格的输出位之间主要比养到几级（评分只占小权重，2026-10-14）
        key = lambda u: (2.0 * math.log1p(u.level) + 0.3 * rating(u.name) + (1.5 if u.awakened else 0.0))
        best = max(cands, key=key)
        cur = s.unit(s.carry) if s.carry else None
        if cur is None:
            s.carry = best.name
        elif best is not cur and key(best) >= key(cur) + 1.5 and (best.awakened or best.level >= 0.6 * cur.level):
            # 新人明显更好、而且等级没低太多才换（换的代价：枫叶舞要换绑、攒的等级不跟过去）。
            # 原来只比"评分 + 0.4×log 等级"，第 13 回合会把 70 多级的虞姬换成 19 级的姜子牙（2026-10-14 流派对比里发现）
            s.carry = best.name
        c = s.unit(s.carry)
        if c and s.maple != c.name:
            old = s.unit(s.maple) if s.maple else None
            if old is None:
                self.e.bind_maple(c)
            elif rating(old.name) < 4 or quality(old.name) == 1:
                self.e.sell(old)                         # 旧宿主是过渡件：卖掉换绑（红叶染锋在 remove 里结算）
                self.e.bind_maple(c)

    # ------------------------------------------------------------ 手牌
    def play_hand(self) -> None:
        e, s = self.e, self.e.s
        for name in list(s.hand):
            if name not in s.hand:
                continue
            cat = category(name)
            if cat in ("英雄牌", "召唤物"):
                if not e.can_play_hero():                   # 透支：回合开始能量为负，本回合不能打英雄牌
                    continue
                u = s.unit(name)
                if u is not None:
                    s.hand.remove(name)
                    e.merge(u)
                elif s.pop_used() < e.pop_cap():
                    s.hand.remove(name)
                    e.deploy(name, cell=self.cell_for(name))
                else:
                    weak = [x for x in s.heroes() if x.pop and x.name != s.carry] if s.round != self.no_swap_round else []
                    if weak and rating(name) > max(rating(w.name) for w in weak) + 2:
                        v = min(weak, key=lambda x: (rating(x.name), x.level))
                        e.sell(v)
                        s.hand.remove(name)
                        e.deploy(name, cell=self.cell_for(name))
                continue
            if name in HOLD or cat in ("棋手技能牌",):
                continue
            targets = [s.unit(s.carry)] if s.carry and s.unit(s.carry) else []
            targets += sorted([x for x in s.heroes() if x.name != s.carry], key=lambda x: (-rating(x.name), -x.level))
            if cat == "装备牌":
                targets = [t for t in targets if len(t.equips) < B.MAX_EQUIPS and e.can_wear(t, name)]
                targets = _by_item_fit(name, targets, s.carry)
            if "古币" in name:
                # 古币机在场：古币打给它（线上打法 2026-10-12：主C 是"每对他使用效果牌降法力上限"的，每回合先吃一张，其余全给古币机）
                mach = next((x for x in s.units if x.name == "古币机"), None)
                if mach is not None:
                    cu = s.unit(s.carry) if s.carry else None
                    fed = s.flags.get("carry_coin_round") == s.round
                    if cu is not None and not fed and "每对" in card_text(cu.name) and "效果牌" in card_text(cu.name):
                        s.flags["carry_coin_round"] = s.round
                        targets = [cu] + [mach]
                    else:
                        targets = [mach] + targets
            if _REMOVE_RE.search(card_text(name)):         # 变身 / 移除牌（化铁为金、限制性进化、体面离场…）：不打要保护的人，打最不值钱的
                keep = self.protected()
                targets = sorted([t for t in targets if t.name not in keep], key=lambda x: (rating(x.name), x.level))
            c = impl(name)
            if not c or not hasattr(c, "on_use"):
                if cat == "装备牌":
                    for t in targets:                       # 穿装备走引擎（件数 / 唯一 / 穿戴限制 / 四选一铸造）
                        if e.equip(t, name):
                            s.hand.remove(name)
                            break
                continue
            done = False
            # 天赋送的牌（和天赋同名，如限制性进化 / 复印机）从手里打出时必须有目标：不指定目标 = 又选了一次天赋、再送一批，会无限循环
            for t in targets + ([] if cat == "天赋牌" else [None]):
                if e.use_card(name, t) is not False:
                    done = True
                    break
            if done and name in s.hand:
                s.hand.remove(name)

    # ------------------------------------------------------------ 准备阶段
    def prep(self) -> None:
        e, s = self.e, self.e.s
        tgt = B.meta.level_target(s.round)
        lv0 = s.lord_level
        while s.lord_level < min(tgt, B.MAX_LORD_LEVEL):
            cost = B.level_cost(s.lord_level, s.xp)
            if not cost or s.energy < cost or not e.spend(cost, src="升级"):
                break
            s.xp += cost
            e.level_ups()
        for _ in range(s.lord_level - lv0):
            self.event("talent")
        # N 选 1 / 枫叶舞强化不再按日志频率额外加：实测（149 局）N 选 1 每局 5.2 次 ≈ 用掉的锦囊 / 兑换券 / 征召令 4.9 张，
        # 枫叶舞强化 2.5 次 ≈ 打出的翩跹舞步 2.7 张 —— 它们都是牌的效果，推演里这些牌自己会弹选择，再按频率加一遍是重复计算，
        # 而且把我们自己的打法（合成多少、击杀多少）当成了游戏规则（2026-10-14）。留下的事件只有规则触发的：升级给天赋、第 3/7/11 回合拍卖。
        views = 0
        while True:
            shop = e.shop_view()
            for n in shop:
                if n and category(n) != "英雄牌":
                    self.buy_card(n)
                    continue
                if not n or s.energy < B.HERO_COST + s.shop_extra_cost:
                    continue
                on = s.unit(n) is not None
                fill = not on and s.pop_used() < e.pop_cap()
                good = rating(n) >= max([rating(x.name) for x in s.heroes()] + [0]) + 2
                if on or fill or good or n == s.carry:
                    if not on and not fill and len(s.hand) >= B.HAND_LIMIT:
                        continue
                    e.buy(n)                                 # 扣费 + 牌库 + 进手牌，发 buy 事件
            self.play_hand()
            self.pick_carry()
            views += 1
            if views > 8 or s.energy < B.REFRESH_COST + B.HERO_COST:
                break
            if s.lord_level < tgt:
                break
            if not e.refresh_shop():                         # 免费刷新 / 本回合免刷新费 / 花 REFRESH_COST，发 refresh 事件
                break
        self.play_hand()
        self.pick_carry()

    def event(self, kind: str) -> None:
        if not self.events:
            return
        R = self.e.s.round
        lst = None
        if kind == "talent":                     # 天赋按棋手等级抽（同等级同品阶；6 级有专属天赋）
            byl = (event_offers().get("offers_by_level") or {}).get(kind) or {}
            lst = byl.get(str(self.e.s.lord_level - 1))   # 记录里的等级是升级前的读数（"5 级"那档几乎全是 6 级专属天赋）
        if not lst:
            offs = (event_offers().get("offers") or {}).get(kind) or {}
            for d in (0, 1, -1, 2, -2):
                lst = offs.get(str(R + d))
                if lst:
                    break
        if not lst:
            return
        opts = self.e.rng.choice(lst)["opts"]
        pick = self.choose(self.e, kind, opts)
        if kind == "talent" and pick and not self.e.s.flags.get("天赋刷新_used") and self.lookahead > 0:
            vals = getattr(self, "_last_vals", {}) or {}
            skip, top = vals.get(None), max((v for k, v in vals.items() if k is not None), default=None)
            if skip is not None and top is not None and top <= skip * 1.005:
                # 三个都几乎没用 → 用掉本局唯一一次天赋刷新（官方：白银段位以上每局 1 次）
                self.e.s.flags["天赋刷新_used"] = True
                opts = self.e.rng.choice(lst)["opts"]
                pick = self.choose(self.e, kind, opts)
                self.e.note("天赋刷新", round=R)
        if kind == "auction":                       # 拍卖要出价，钱不够就分到剩下的（engine.auction）
            pick = self.e.auction(pick, opts)
        if pick:
            self._take(kind, pick)
            self.e.note("事件", kind=kind, pick=pick, opts=opts)

    # ------------------------------------------------------------ 一回合
    def round(self) -> dict:
        e, s = self.e, self.e.s
        self.prep()
        self.arrange()
        e.battle()
        if self.events and s.round in (3, 7, 11):
            self.event("auction")
        lv0 = s.lord_level
        e.round_end()
        for _ in range(s.lord_level - lv0):
            self.event("talent")
        e.round_start()
        self.play_hand()
        c = s.unit(s.carry) if s.carry else None
        return {"round": s.round, "lord": s.lord_level, "energy": s.energy, "carry": s.carry,
                "carry_lv": c.level if c else 0.0, "carry_awk": c.awaken if c else 0,
                "team_lv": sum(u.level for u in s.heroes()), "n": len(s.heroes()), "merges": s.merges,
                "yizhi": "遗志" in s.hand}


def simulate(st: State, n: int = 50, rounds_to: int = 14, seed: int = 0, **kw) -> list:
    out = []
    for i in range(n):
        s2 = st.clone()
        e = Engine(s2, random.Random(seed * 100003 + i))
        p = Player(e, **kw)
        traj = []
        while s2.round < rounds_to:
            traj.append(p.round())
        out.append(traj)
    return out


def summarize(runs: list, rounds=(5, 8, 10, 12, 14)) -> dict:
    import statistics as S
    out = {}
    for R in rounds:
        rows = [next((x for x in tr if x["round"] == R), None) for tr in runs]
        rows = [x for x in rows if x]
        if rows:
            out[R] = {"carry_lv": S.median(x["carry_lv"] for x in rows), "awk": sum(x["carry_awk"] >= 3 for x in rows) / len(rows),
                      "lord": S.median(x["lord"] for x in rows), "team_lv": S.median(x["team_lv"] for x in rows),
                      "merges": S.median(x["merges"] for x in rows), "遗志在手": sum(x["yizhi"] for x in rows) / len(rows)}
    return out
