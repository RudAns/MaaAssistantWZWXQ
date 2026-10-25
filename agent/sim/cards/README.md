# 写卡规范（agent/sim/cards）

每张卡 = 一个类，用 `@card("卡名")` 注册，**卡面写什么就做什么**，只做养成相关的事（等级 / 临时等级 / 觉醒 / 装备 / 获得牌 / 能量 / 经验 / 人口 / 召唤 / 移除 / 触发别人的效果）。
战斗数值（攻击、伤害、护盾、法力、攻速、控制）**不实现**：在类的 docstring 里写一句"战斗效果：…，不影响养成"即可（以后要用再补）。

**引擎已经有的东西一律走引擎接口，不要在批次模块里再写一份**（2026-10-14 把 8 个批次各自补的辅助函数统一进了 engine.py / state.py）。
引擎里确实没有的，先在本模块写，并在 docstring 写"接口缺口"，之后再统一进引擎。

## 钩子（engine.py 开头有完整列表）

```python
@card("李元芳")
class 李元芳:
    """整备：对距离最近的【河洛】英雄使用1张古币。"""
    def on_prepare(self, eng, u):
        t = eng.nearest(u, lambda x: eng.is_faction(x, "河洛"))
        if t:
            eng.coin(t[0], 1, src="李元芳")
```

- 自身时机：`on_deploy / on_prepare / on_open / on_battle_start / on_kill(n=) / on_death / on_sacrifice / on_win / on_lose / on_sell / on_merge`，参数 `(eng, u)`，u = 这个单位。
  **觉醒翻倍由引擎处理（执行两次）**；卡面另有觉醒写法的，类里设 `awaken_custom = True` 自己处理。
- 持有牌（天赋、装备、棋手牌、在场光环、手牌里 `in_hand = True` 的牌、本回合打出过的 `round_cards`）：
  `on_round_start(eng, holder)`、`on_round_end(eng, holder)`（回合结束、**收入到账之前**）、
  `on_battle_begin(eng, holder)`（整场战斗开始一次，在核心分配和开团之前；"开战前"类效果放这里）、
  `on_battle_end(eng, holder, win=)`；holder = 挂靠单位（天赋 / 棋手牌 / 手牌为 None）。
- 装备：挂在单位身上，`on_round_start(eng, holder)` 等的 holder = 穿戴者；穿上时 `on_equip(eng, u)`（由 `eng.equip` 调）。
- 使用（效果牌 / 天赋牌 / 赠送牌）：`on_use(eng, target)`，target 可为 None；不合法返回 False（例：古币指向非河洛）。
- 反应器（"每当…时"）：`react_<事件>(eng, holder, **kw)`。**反应器要自己判断 holder 是否在场**。事件和参数：

  | 事件 | 参数 | 谁发 |
  |---|---|---|
  | deploy / prepare / open / battle_start / kill / death / sacrifice / win / lose / merge / sell | `unit`、`triggered`（kill 另有 `n`） | `eng.fire` |
  | effect_card | `card, target` | `eng.use_card`（效果牌） |
  | coin | `target, fine, active`（active = 从手牌打出，李元芳这类被动使用为 False） | `eng.coin` |
  | temp | `unit, n, src` | `eng.temp`（司空震） |
  | summon | `unit` | `eng.summon`（往生图腾吸收） |
  | equip | `unit, card` | `eng.equip` |
  | core_up | `n, src` | `eng.core_add`（海诺） |
  | gain | `card` | `eng.gain_card` |
  | buy | `card, cost` | `eng.buy` |
  | spend | `n, src` | `eng.spend`（消费返利） |
  | refresh | — | `eng.refresh_shop`（节能商店、神秘英雄牌） |
  | sell_card | `card` | `eng.sell_card`（丝绸之路、大买卖） |
  | eliminated | `n` | `eng.eliminate`（拾荒者、资源再分配） |
  | round_end | — | `eng.round_end` |
  | lord_levelup | `level` | `eng.level_ups` |
  | thaw | `unit` | `batch_07.thaw`（冰之心解冻入口） |

  **钩子只会收到签名里声明过的关键字参数**（引擎按签名过滤），所以引擎给事件加参数不会弄坏旧卡；需要新参数就在签名里加上。
- 被动加成 / 修正：`coin_bonus(eng, holder, target, fine[, active]) -> int`（古币额外等级）、`grant_keywords(eng, holder)`（每回合开始重新赋予关键词：钟馗的牺牲、始祖熊灵的大河）、
  `gain_redirect(eng, holder, name) -> list|None`（获得牌时改成别的牌：零钱兑换、演武）、`xp_need_delta(eng, holder) -> int`（升级所需经验修正：人才济济）。
- 类属性：`in_hand = True`（拿在手里就生效）、`awaken_custom`、`core_no_share`（持有时核心不开战分配：能量改造 / 光棱塔）、
  `core_direct_cap`（定向传输 200）、`blocks_core`（装备：穿戴者不参与核心分配，星源星徽）、`no_attack`（召唤物无法攻击，不分击杀：故障机器人）。

## 引擎操作

- 等级：`eng.level(u, n, src)` 永久等级；`eng.temp(u, n, src)` 临时等级（发 temp）；`eng.coin(t, n, src, fine, active)`；`eng.steal(u, src)` 夺取（周围 1 格随机 1 名，最多 10 级）。
- 用牌 / 手牌：`eng.use_card(name, target)`；`eng.gain_card(name, **meta)` 进手牌（满 10 张丢掉；meta 记进手牌状态，如 `transient=True`）；
  `eng.hand_meta(name)` / `eng.take_meta(name)`（同名按获得先后，先拿到的先用；每张都有 `got` = 获得回合）；`eng.hand_remove(name)`；`eng.sell_card(name)`（+1 能量）；
  `eng.attach_round(name)` 本回合整回合生效的牌（锦囊）。
- 上场 / 合成 / 离场：`eng.deploy / merge / remove / sell / bind_maple`；`eng.summon(name, near, level, battle_only=False)`（发 summon；`battle_only=True` 的战斗结束由引擎移除）；
  `eng.rename(u, new, src)` 变身（主C / 枫叶舞记的名字一起改）。
- 触发：`eng.fire(u, timing)`；**"触发其登场 / 整备效果"用 `eng.trigger(u, timing)`**（= `fire(..., triggered=True)`）；
  "每当你使用 / 打出英雄牌时"的卡用 **`eng.is_card_use()`** 区分（上场 / 合成 / 觉醒回手再打出 = True，被唤醒 / 曹操 / 临阵磨枪 触发的 = False）。
- 关键词：`has_effect(name, kw)`（= `has_keyword`：卡面有"kw："条目或第一行单独列出 kw；**只是"提到"的不算**——李信 / 大司命不拥有牺牲，张良 / 朵莉亚 / 大乔 / 小乔不拥有开团 / 整备 / 登场）；
  `mentions_keyword(name, kw)`（只要提到就算）；`eng.has_kw(u, kw)`（含被赋予的 u.keywords）；`eng.unit_has_effect(u, kw)`（本体或身上装备）。
- 装备：`eng.equip(u, name, bound=False, forge=True, force=False)` —— 3 件上限、"唯一"同名一人一件、穿戴限制（`eng.can_wear`：读卡面"限定远程 / 近战英雄""仅三分英雄""仅1-3阶英雄""可穿戴给【X】英雄"，也认类的 `can_equip / wearable / wear_faction`）、
  基础装备当场四选一铸造（`data/equip_recipes.json`，`eng.choose("forge", 成品…)`）、`bound=True` 或卡面"无法拆卸"进 `u.bound`、调 `on_equip`、发 equip；`force=True` 用于出生自带。
  【X】专属装备不限制穿戴，专属效果是否生效用 `eng.exclusive_active(u, name)`（全域共鸣）。
- 阿科米亚核心：`eng.s.core`；`eng.core_add(n, src, event=True)`（n 可负，最低 0；提升时发 core_up）；开战分配 `eng.core_distribute()` 由引擎在 `battle()` 开头统一做——
  参与者 = 日落海英雄或带"参与阿科米亚核心"关键词的（阿科米亚之子），排除 `blocks_core` 装备的穿戴者；每人 floor(min(核心/人数, 核心×50%))；`core_direct_cap` 先给穿装备最多的那名。**卡里不要再自己分配**。
- 觉醒：`eng.awaken_step(u, n, src)`（非合成途径的觉醒进度，封顶 3；凑满 = 回手再打出一次登场）。
- 经济：`eng.spend(n, src)`（认 `s.energy_floor`，发 spend）；`eng.buy(name)`（扣 HERO_COST + `s.shop_extra_cost`、记牌库、进手牌，发 buy）；
  `eng.refresh_shop()`（`s.free_refresh_all` > `s.free_refresh` > 花 REFRESH_COST，发 refresh）；`eng.can_play_hero()`（透支）；`eng.add_xp(n, src)`；`eng.xp_need()`。
- 棋手：`eng.s.hp` / `eng.add_hp(delta, set_to)`（上限 60，`s.hp_guard` = 降到 0 保留 1 点的次数，一线生机）；`eng.s.rank`（None = 不知道）；`eng.eliminate(n)`；
  `eng.add_lord_card(name)`、`eng.pick_lord_boost(k=3)`（为枫叶舞挑强化牌，翩跹舞步一族）。
- 战斗估计：常量 `BATTLE_SECONDS = 20`、`LORD_HP_MAX = 60`；`eng.attacks(u)`（普攻次数 = 攻速 × 20 秒）、`eng.casts(u)`（放技能次数，`u.counters["casts"]` 可直接指定）；
  `eng.survivors()`（最近一场战斗活下来的英雄，复生后活下来的算），`s.battle_dead` / `s.battle_deaths`。
- 选择与随机：`eng.nearest(u, pred, k)`、`eng.pick(list, k)`、`eng.lowest / highest(list, k)`；`eng.draw_hero(tier, fac)` 按牌库剩余抽英雄、`eng.take(name)` 记牌库被拿走；
  `eng.choose(kind, options, **ctx)` 让推演打法选（N 选 1、兑换券、四选一…）。

## 局面（State）字段

`round / lord_level / xp / energy / energy_floor / pop_bonus / units / hand / hand_meta / shop / free_refresh / free_refresh_all / shop_extra_cost /
talents / lord_cards / round_cards / maple / maple_keep / carry / core / hp / hp_guard / rank / eliminated / battle_dead / battle_deaths / pool_taken / merges / flags / log`。
`flags` 只放卡面自己的计数（"限一次"已用、本回合计数…；`round:` 前缀回合开始清）——**上面这些有正式字段的量不要再放 flags**。

## 战斗顺序（engine.battle）

整场开始 `on_battle_begin` → 核心分配 → 开团 → 战斗开始 → 击杀（`no_attack` 的不分）→ 阵亡 / 牺牲（复生）→ 凯旋 / 败阵 → `on_battle_end` → 移除 `battle_only` 召唤物 → 临时等级清零。
回合：`round_start`（清 `round:` 标记、`free_refresh_all`、`round_cards` → 赋予关键词 → 持有牌 on_round_start → 整备）；`round_end`（on_round_end / round_end 事件 → 经验 → 回合 +1 → 收入 → 升级）。

## 卡面歧义

写不清的地方（结算时机、取整、随机范围、是否包含自己…）按最自然的读法实现，并在 docstring 里写 `待核对：…`。**不要猜一个数值硬塞**。

## 测试

每张卡的测试由**另一个人**照卡面独立写（tools/sim/test_batch_*.py）：搭一个最小局面 → 执行 → 断言卡面说的结果。
跨卡组合与整局随机跑见 `tools/sim/test_integration.py`。改引擎后 8 个批次测试 + 集成测试都要跑：

```bash
for i in 0 1 2 3 4 5 6 7; do .venv/Scripts/python.exe -X utf8 tools/sim/test_batch_0$i.py | tail -1; done
.venv/Scripts/python.exe -X utf8 tools/sim/test_integration.py
```
