"""每局特征抽取（2026-10-02：从前 3 名和 5、6 名的对局里找规律）。

  python tools/analysis/match_features.py            # 默认第 49~85 局
  python tools/analysis/match_features.py 60 85      # 指定局号范围

读 data/logs/agent_one_<N>.log + 对应 live_*/{states,ledger,laya_points}.jsonl，
每局一行写 data/analysis/match_features.jsonl。只读日志，不改任何东西。
没有"本局名次"的局（还在打/中断）跳过。

口径说明（都是"尽量从日志里还原"，有误差）：
- 主C：states.jsonl 每帧的 "carry"；最终主C = 最后一个非空值。
- 主C 等级：states 里 board[主C].lv（rules.board_facts，读不到为 None），取该回合准备阶段最后一个读得到的值。
- 主C 到手回合：最终主C 第一次出现在账本「获得卡牌」/ 手牌 / 场上 的回合。
- 对主C 用牌：账本「用牌」目标 == 当时的主C（逐帧主C）；装备另计。
- 装备品阶：cards.json quality；四选一成功时按日志「选第 i 个「成品」」换成成品名。
- 买牌：按账本「买牌」计（states.jsonl 里同一次买入会被记成多帧 executed，不能用）；刷新/升级/N选1 按 states 的已执行动作。
- 剩余能量：每回合准备阶段最后一帧的 energy（回合末没花掉的钱）。
- 名次结果有幸存者偏差：活得久的局整局计数（卡牌数、刷新数…）天然更多，比较要看同一回合窗口（acts_by_round）。
- 拍卖拍到：拍卖回合出过价的英雄，在本回合/下回合账本「获得卡牌」里出现 → 记为拍到（启发式）。
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "data" / "logs"
OUT = ROOT / "data" / "analysis" / "match_features.jsonl"
CARDS = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))["cards"]


def q_of(name):
    c = CARDS.get(name)
    return c.get("quality") if c else None


def cat_of(name):
    c = CARDS.get(name)
    return c.get("category") if c else None


def jl(path):
    if not path.exists():
        return []
    out = []
    for line in path.open(encoding="utf-8", errors="replace"):
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def first_round_at_least(per_round_level, k):
    for r in sorted(per_round_level):
        if per_round_level[r] is not None and per_round_level[r] >= k:
            return r
    return None


def analyze(n: int):
    logp = LOGS / f"agent_one_{n}.log"
    if not logp.exists():
        return None
    log = logp.read_text(encoding="utf-8", errors="replace")
    rank = re.findall(r"本局名次：第 *(\d) *名", log)
    if not rank:
        return None
    runs = re.findall(r"live_\d+", log)
    run = LOGS / runs[0]
    states = jl(run / "states.jsonl")
    ledger = jl(run / "ledger.jsonl")
    points = jl(run / "laya_points.jsonl")

    f = {"match": n, "rank": int(rank[-1]), "run": run.name}

    # ---------- states 逐帧 ----------
    ban = Counter()
    carry_seq = []                      # [(round, carry)]
    lv_by_round = defaultdict(dict)     # r -> {carry_name: lv}（准备阶段最后读数）
    board_tot = {}                      # r -> 全场等级和
    board_n = {}                        # r -> 场上人数（账本）
    eq_max = defaultdict(int)           # 英雄 -> 看到过的最多装备件数
    lord_lv = defaultdict(list)         # r -> [level 读数]
    energy_end = {}                     # r -> 准备阶段最后一帧能量
    pop_cap = {}
    seen_carry_round = {}               # 英雄 -> 第一次在手牌/场上出现的回合
    acts = Counter()
    acts_r = Counter()                  # (type, round) -> 次数
    talents = {}                        # r -> name（同回合重试取最后）
    boosts = {}
    bids = defaultdict(set)             # r -> {拍品}
    merges = 0
    cur_carry_at = {}                   # r -> 该回合最后的主C
    rframes = Counter()                 # r -> 帧数（回合号偶尔误读成 99，按帧数过滤）
    for x in states:
        s = x.get("state") or {}
        r = s.get("round_no") or 0
        if not r:
            continue
        rframes[r] += 1
        if s.get("banned_faction"):
            ban[s["banned_faction"]] += 1
        ph = s.get("phase")
        c = x.get("carry")
        if c is not None:
            if c and (not carry_seq or carry_seq[-1][1] != c):
                carry_seq.append((r, c))
            if c:
                cur_carry_at[r] = c
        if ph == "prepare":
            if isinstance(s.get("level"), int) and s["level"] > 0:
                lord_lv[r].append(s["level"])
            if isinstance(s.get("energy"), int) and 0 <= s["energy"] < 100:
                energy_end[r] = s["energy"]
            if s.get("population_valid") and s.get("population_cap"):
                pop_cap[r] = s["population_cap"]
            for h in s.get("hand") or []:
                seen_carry_round.setdefault(h, r)
        b = x.get("board")
        if isinstance(b, dict) and b:
            for nm, v in b.items():
                seen_carry_round.setdefault(nm, r)
                if not isinstance(v, dict):
                    continue
                if isinstance(v.get("lv"), int):
                    lv_by_round[r][nm] = v["lv"]
                if isinstance(v.get("eq"), int):
                    eq_max[nm] = max(eq_max[nm], v["eq"])
            if ph == "prepare":
                tot = sum(v.get("lv") for v in b.values() if isinstance(v, dict) and isinstance(v.get("lv"), int))
                if tot:
                    board_tot[r] = tot
                board_n[r] = len(b)
        if x.get("executed") and x.get("actions"):
            a = x["actions"][0]
            t = a.get("type")
            acts[t] += 1
            if t != "buy_piece":            # 买牌按账本计（states 里同一次买入会被记多帧）
                acts_r[(t, r)] += 1
            if t == "talent_pick" and a.get("name"):
                talents[r] = a["name"]
            elif t == "lord_boost_pick" and a.get("name"):
                boosts[r] = a["name"]
            elif t == "auction_bid" and a.get("piece") and not a.get("select_only"):
                bids[r].add(a["piece"])
            elif t == "drag_card" and a.get("merge"):
                merges += 1
                acts_r[("merge", r)] += 1

    f["ban"] = ban.most_common(1)[0][0] if ban else ""
    end_round = max((r for r, k in rframes.items() if k >= 5 and r <= 30), default=0)
    f["end_round"] = end_round
    f["carry_seq"] = carry_seq
    f["first_carry"] = carry_seq[0][1] if carry_seq else ""
    f["final_carry"] = carry_seq[-1][1] if carry_seq else ""
    f["carry_changes"] = max(0, len(carry_seq) - 1)
    f["carry_set_round"] = carry_seq[0][0] if carry_seq else None
    fc = f["final_carry"]

    # ---------- 账本 ----------
    routes, backups = [], []
    gained = Counter()
    gained_by_round = defaultdict(list)
    to_carry = []        # (round, card, category)
    to_final = []
    equips_final = []    # (round, item)
    sells = 0
    swaps = 0
    for e in ledger:
        ev = e.get("event")
        r = e.get("round") or 0
        if ev == "锁定路线":
            routes.append((r, e.get("阵容"), e.get("主C"), e.get("来源")))
        elif ev == "换备选路线":
            backups.append((r, e.get("原"), e.get("新"), e.get("原因")))
        elif ev == "获得卡牌":
            gained[e.get("类别") or cat_of(e.get("牌")) or "?"] += 1
            gained_by_round[r].append(e.get("牌"))
        elif ev == "用牌":
            tgt, card, cat = e.get("目标"), e.get("牌"), e.get("类别")
            cc = cur_carry_at.get(r)
            if tgt and cc and tgt == cc:
                to_carry.append((r, card, cat))
            if tgt and tgt == fc:
                to_final.append((r, card, cat))
                if cat == "装备牌":
                    equips_final.append((r, card))
        elif ev == "买牌":
            acts_r[("buy_piece", r)] += 1
            acts_r[("buy_hero" if cat_of(e.get("牌")) == "英雄牌" else "buy_other", r)] += 1
        elif ev in ("卖出", "卖手牌"):
            sells += 1
        elif ev == "换阵容卖人":
            swaps += 1
    f["first_route"] = routes[0][1] if routes else ""
    f["final_route"] = routes[-1][1] if routes else ""
    f["route_carry"] = routes[-1][2] if routes else ""
    f["backup_switches"] = backups
    f["cards_gained"] = sum(gained.values())
    f["cards_gained_by_cat"] = dict(gained)
    f["cards_gained_R1_8"] = sum(len(v) for k, v in gained_by_round.items() if 1 <= k <= 8)
    f["sells"] = sells
    eff = [t for t in to_carry if t[2] != "装备牌"]
    f["cards_to_carry"] = len(eff)
    f["cards_to_carry_R5_10"] = sum(1 for t in eff if 5 <= t[0] <= 10)
    f["cards_to_carry_by_round"] = dict(Counter(t[0] for t in eff))
    f["cards_to_carry_names"] = [t[1] for t in eff]
    f["equips_to_carry"] = sum(1 for t in to_carry if t[2] == "装备牌")

    # 装备：四选一铸出的成品（日志）
    forged = []    # (英雄, 成品)
    last_pick, last_hero = None, None
    for line in log.splitlines():
        m = re.search(r"选第 \d 个「([^」]+)」", line)
        if m:
            last_pick = m.group(1)
        m = re.search(r"基础装备 \S+ 穿给 ([^(（\s]+)", line)
        if m:
            last_hero = m.group(1)
        if "【装备四选一】✅" in line and last_pick and last_hero:
            forged.append((last_hero, last_pick))
            last_pick = None
    f_items = [p for h, p in forged if h == fc]
    base_ct = sum(1 for _, it in equips_final if q_of(it) == 1)
    items = f_items + [it for _, it in equips_final if q_of(it) != 1]
    items += ["(基础未铸)"] * max(0, base_ct - len(f_items))
    f["carry_items"] = items
    f["carry_item_q"] = [q_of(i) for i in items]
    f["carry_eq_seen_max"] = eq_max.get(fc, 0)
    f["carry_items_n"] = max(len(items), eq_max.get(fc, 0))
    f["carry_items_q2plus"] = sum(1 for q in f["carry_item_q"] if q and q >= 2)
    f["forged_total"] = len(forged)
    f["forged_to_carry"] = len(f_items)

    # ---------- 主C 等级 ----------
    def lv_at(r, who=None):
        for rr in range(r, 0, -1):          # 读不到往前找一回合（最多 1 回合）
            d = lv_by_round.get(rr) or {}
            w = who or cur_carry_at.get(rr)
            if w in d:
                return d[w]
            if rr <= r - 1:
                break
        return None
    for r in (5, 8, 10, 12):
        f[f"carry_lv_R{r}"] = lv_at(r)
        f[f"final_carry_lv_R{r}"] = lv_at(r, fc) if fc else None
        f[f"board_lv_R{r}"] = board_tot.get(r)
        f[f"board_n_R{r}"] = board_n.get(r)
    f["carry_lv_max"] = max((d.get(fc) for d in lv_by_round.values() if isinstance(d.get(fc), int)), default=None)
    arr = None
    if fc:
        cands = [seen_carry_round.get(fc)]
        cands += [r for r, v in gained_by_round.items() if fc in v and r]
        cands = [c for c in cands if c]
        arr = min(cands) if cands else None
    f["carry_arrive_round"] = arr
    f["carry_arrive_lv"] = None
    if arr:
        for r in range(arr, arr + 3):
            if fc in (lv_by_round.get(r) or {}):
                f["carry_arrive_lv"] = lv_by_round[r][fc]
                break

    # ---------- 经济 ----------
    f["buys"] = sum(v for (tt, _), v in acts_r.items() if tt == "buy_piece")
    f["refreshes"] = acts["refresh_shop"]
    f["levelups"] = acts["buy_xp"]
    f["card_picks"] = acts["card_pick"]
    f["merges"] = merges
    for t, key in (("buy_piece", "buys"), ("refresh_shop", "refreshes"), ("buy_xp", "levelups"),
                   ("card_pick", "card_picks")):
        f[key + "_R1_8"] = sum(v for (tt, r), v in acts_r.items() if tt == t and 1 <= r <= 8)
    f["merges_R1_8"] = sum(v for (tt, r), v in acts_r.items() if tt == "merge" and 1 <= r <= 8)
    f["cards_to_carry_R1_8"] = sum(1 for t in eff if t[0] <= 8)
    f["coin_to_carry"] = sum(1 for t in eff if t[1] == "古币")
    f["noncoin_to_carry_R5_10"] = sum(1 for t in eff if 5 <= t[0] <= 10 and t[1] != "古币")
    f["energy_end"] = {r: energy_end[r] for r in sorted(energy_end)}
    left = [energy_end[r] for r in energy_end if 2 <= r <= 10]
    f["energy_left_mean_R2_10"] = round(statistics.mean(left), 2) if left else None
    f["energy_left_sum_R2_10"] = sum(left) if left else None
    lv_med = {r: int(statistics.median(v)) for r, v in lord_lv.items() if v}
    # 等级只增不减：做一次前缀 max 抹掉单帧误读
    run_max, lv_mono = 0, {}
    for r in sorted(lv_med):
        run_max = max(run_max, lv_med[r])
        lv_mono[r] = run_max
    for k in (3, 4, 5, 6):
        f[f"lord_lv{k}_round"] = first_round_at_least(lv_mono, k)
    f["lord_lv_by_round"] = lv_mono
    f["pop_cap_max"] = max(pop_cap.values()) if pop_cap else None

    # ---------- 天赋 / 强化 / 拍卖 ----------
    f["acts_by_round"] = {t: {r: v for (tt, r), v in sorted(acts_r.items()) if tt == t}
                          for t in sorted({tt for tt, _ in acts_r})}
    f["gained_by_round_n"] = {r: len(v) for r, v in sorted(gained_by_round.items()) if r}
    f["talents_r"] = [[r, talents[r]] for r in sorted(talents)]
    f["boosts_r"] = [[r, boosts[r]] for r in sorted(boosts)]
    f["talents"] = [talents[r] for r in sorted(talents)]
    f["talent_rounds"] = sorted(talents)
    f["lord_boosts"] = [boosts[r] for r in sorted(boosts)]
    won = []
    for r, ps in bids.items():
        got = set(gained_by_round.get(r, [])) | set(gained_by_round.get(r + 1, []))
        hit = sorted(ps & got)
        if hit:
            won.append((r, hit))
    f["auction_rounds_bid"] = sorted(bids)
    f["auction_won"] = won
    f["auction_won_n"] = len(won)

    # ---------- Laya / 局面网络 ----------
    kinds = Counter((p.get("kind"), p.get("model") or "laya") for p in points)
    f["answers"] = {f"{k}:{m}": v for (k, m), v in sorted(kinds.items())}
    f["answers_used_rule"] = sum(1 for p in points if str(p.get("used")) not in ("laya", "cgnet"))
    return f


def main(argv):
    lo, hi = (int(argv[1]), int(argv[2])) if len(argv) >= 3 else (49, 85)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in range(lo, hi + 1):
        try:
            f = analyze(n)
        except Exception as ex:   # noqa: BLE001
            print(f"第 {n} 局失败: {ex!r}")
            continue
        if f is None:
            print(f"第 {n} 局：无名次，跳过")
            continue
        rows.append(f)
        print(f"第 {n:>2} 局 第{f['rank']}名 禁{f['ban'] or '?':<4} 主C {f['final_carry'] or '-':<5} 到手R{f['carry_arrive_round']} "
              f"lvR5/8/10 {f['carry_lv_R5']}/{f['carry_lv_R8']}/{f['carry_lv_R10']} 用牌给主C {f['cards_to_carry']} 卡 {f['cards_gained']}")
    with OUT.open("w", encoding="utf-8") as fp:
        for f in rows:
            fp.write(json.dumps(f, ensure_ascii=False) + "\n")
    print(f"写入 {OUT}（{len(rows)} 局）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
