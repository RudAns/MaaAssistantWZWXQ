"""感知层离线回归测试：用对局录像帧驱动 StateExtractor，校验阶段识别与字段提取。

用法: python -X utf8 tools/test_perception_offline.py [帧目录]
需要 MuMu 模拟器在线（仅为绑定 Tasker；识别吃的是录像帧，不是实时画面）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import cv2  # noqa: E402

from maa.pipeline import JOCR, JRecognitionType  # noqa: E402

from agent.adb_client import connect_adb_controller  # noqa: E402
from agent.perception.extractors import StateExtractor  # noqa: E402
from agent.win32_client import create_tasker  # noqa: E402


class _Box:
    def __init__(self, text: str, box):
        self.text = text
        self.box = box


class _RecoDetail:
    """模拟 Context.run_recognition 的返回结构（hit / best_detail / all_results）。"""

    def __init__(self, task_detail):
        reco = None
        if task_detail is not None:
            for node in getattr(task_detail, "nodes", []) or []:
                reco = node.recognition
                if reco is not None:
                    break
        self.hit = bool(getattr(reco, "hit", False)) if reco else False
        best = getattr(reco, "best_result", None) if reco else None
        self.best_detail = _Box(best.text, best.box) if best else None
        results = getattr(reco, "filtered_results", None) or getattr(reco, "all_results", None) or []
        self.all_results = [_Box(r.text, r.box) for r in results]


class FrameContext:
    """把 context.run_recognition(entry, image, pipeline_override) 桥接到
    Tasker.post_recognition(JOCR, image) —— 离线喂录像帧用。"""

    def __init__(self, tasker):
        self.tasker = tasker

    def run_recognition(self, entry, image, pipeline_override=None):
        node = (pipeline_override or {}).get(entry, {})
        param = JOCR(roi=tuple(node.get("roi", (0, 0, 0, 0))))
        if node.get("expected"):
            param.expected = [node["expected"]]
        job = self.tasker.post_recognition(JRecognitionType.OCR, param, image)
        return _RecoDetail(job.wait().get())


# 帧 → (期望阶段, 需要检查的字段)
# 除了旧版 UI 的录像帧，再加几张**实战踩坑现场**（data/logs/fixtures/）。
# 这些是"当时真的把 agent 卡死了"的帧，放进回归里，同类问题几秒就能发现，不用开局。
EXTRA_CASES = {
    # 锦囊打出后弹的三选一。它把「准备/开战」按钮盖住 → prepare 判据全不命中，
    # 曾经落进 unknown 一直挂着（2026-09-22 第六局）。现在必须判成 card_pick。
    "card_pick_3choices.png": ("card_pick", ["card_options"]),
    # 棋手详情面板：点棋盘上的棋手本体会弹出，几乎盖满屏幕（x280~1160/y30~560）。
    # 认不出它就整局卡死 —— 原来的逃生点 TAP_SIDE_GROUND(1120,500) 就在面板内部，
    # 点了等于没点（2026-09-22 实测：误点到英雄、进入死循环）。
    "lord_panel.png": ("lord_panel", []),
    # 拍卖**出价阶段**：顶部横幅是「已为最高出价者，不可切换」，
    # 不含"预览/拍卖/开启"任何一个词。2026-09-23 实测这一帧被判成 prepare，
    # 随后 N选1 检测器把 5 个拍品当成候选、连点 3 次（等于对着拍品乱点）。
    "auction_top_bidder.png": ("auction", []),
}

CASES = {
    "frame_0039.png": ("prepare", ["shop", "gold", "hand"]),
    "frame_0062.png": ("prepare", ["shop", "gold", "hand"]),
    "frame_0014.png": ("battle", ["players"]),
    "frame_0026.png": ("auction", ["auction"]),
    "frame_0097.png": ("talent", ["talents"]),
    "frame_0224.png": ("settle", []),
}


# ---- 棋手等级（点数徽章）----
# 1点=1级 … 5点=5级，**6级也是5点但彩色**（2026-09-22 实机确认）。
# 这是纯像素运算（不走 OCR），所以可以脱离模拟器直接断言。
# 样本取自不同对局的真实帧；4 级那张曾经被读成 5 级（中心反光算成了第 5 个点）。
LEVEL_CASES = [
    ("data/logs/live_1789986401/phase_prepare_1789987041.png", 1),
    ("data/logs/live_1789986401/phase_prepare_1789987263.png", 2),
    ("data/logs/live_1789986401/phase_prepare_1789989191.png", 3),
    ("data/logs/live_1789992739/phase_prepare_1789993573.png", 4),
]


def check_levels() -> list[str]:
    from agent.perception.extractors import _imread, read_lord_level
    bad = []
    for rel, want in LEVEL_CASES:
        path = PROJECT_DIR / rel
        if not path.exists():
            print(f"[跳过] {rel} 不存在")
            continue
        got = read_lord_level(_imread(path))
        flag = "OK" if got == want else "FAIL"
        print(f"[{flag}] 棋手等级 {Path(rel).name}: {got}（期望 {want}）")
        if got != want:
            bad.append(f"棋手等级误读 {rel}: {got} != {want}")
    return bad


def check_phase_cache(ctx) -> list[str]:
    """**同一个 extractor 连跑两帧**：阶段缓存会不会把自己毒死。

    2026-09-22 实战：`card_pick` 被写进 _phase_cache，下一帧复验时
    `PHASE_PREDICATES["card_pick"]` KeyError —— 而 card_pick 根本不在那张表里。
    单帧用例永远抓不到这个，必须连着跑两帧。
    """
    from agent.perception.extractors import StateExtractor, _imread
    bad = []
    frame = PROJECT_DIR / "data/logs/fixtures/card_pick_3choices.png"
    if not frame.exists():
        print("[跳过] card_pick fixture 不存在")
        return bad
    img = _imread(frame)
    ex = StateExtractor(PROJECT_DIR / "assets" / "image")
    try:
        first = ex.extract(ctx, img).phase
        second = ex.extract(ctx, img).phase          # 第二帧走缓存复验路径
    except Exception as exc:                          # noqa: BLE001
        print(f"[FAIL] 阶段缓存复验抛异常: {exc!r}")
        return [f"阶段缓存复验抛异常: {exc!r}"]
    ok = first == second == "card_pick"
    print(f"[{'OK' if ok else 'FAIL'}] 阶段缓存连跑两帧: {first} -> {second}（期望 card_pick 两次）")
    if not ok:
        bad.append(f"阶段缓存复验结果不对: {first} -> {second}")

    # **card_pick 之后必须能回到商店**。card_pick 的复验只走 _detect_card_pick，
    # 而它原来只看"卡名查不查得到"——商店那一排卡名同样查得到，于是一旦进过
    # 一次 N选1 就永久卡在 card_pick，对着商店卡面连点（= 盲买）。
    # 这一条单帧测不出来，必须"先喂 N选1 帧、再喂商店帧"。
    # 任何"摆着一排卡"的界面都可能被 card_pick 这个兜底检测器认领，
    # 所以每加一个排除判据，就在这里加一帧对应的真实界面。
    for rel, want in (("data/logs/match_170447/frame_0039.png", "prepare"),
                      ("data/logs/match_170447/frame_0026.png", "auction")):
        frame2 = PROJECT_DIR / rel
        if not frame2.exists():
            continue
        ex2 = StateExtractor(PROJECT_DIR / "assets" / "image")
        ex2.extract(ctx, img)                      # 先把 _phase_cache 喂成 card_pick
        got = ex2.extract(ctx, _imread(frame2)).phase
        ok2 = got == want
        print(f"[{'OK' if ok2 else 'FAIL'}] card_pick 之后喂 {want} 帧: {got}（期望 {want}）")
        if not ok2:
            bad.append(f"card_pick 卡住了，{want} 帧被判成 {got}")
    return bad


def check_hp_filter() -> list[str]:
    """血量判据（纯函数，不用模拟器）。

    2026-09-23 实战：回合4 血 57 → 回合5 直接变 **0**，人还在打。
    根因是原来的"大跌幅等下一帧复核"走的是**像素缓存**分支，
    而"像素没变"恰恰说明还是同一批像素 —— 同一个误读自己给自己背书。
    """
    from agent.perception.extractors import filter_hp
    bad = []
    cases = [
        (57, 0, 57, "0 血还在打 = 误读"),
        (57, 7, 57, "首位数字被吃掉"),
        (60, 57, 57, "正常掉血要采信"),
        (57, 30, 57, "跌幅超上限先不信"),
        (57, 60, 57, "血量只减不增"),
    ]
    for last, hp, want, why in cases:
        got, _ = filter_hp(last, hp, (0, 0))
        ok = got == want
        print(f"[{'OK' if ok else 'FAIL'}] 血量 {last}->{hp} => {got}（{why}）")
        if not ok:
            bad.append(f"血量判据 {why}: {last}->{hp} 给出 {got}，期望 {want}")
    # **"连读就认"要讲时间 + 认错了要能改回来**（2026-09-23 实战：52 被持续读成 2，
    # 连读 3 次后认了；之后真实的 52 全被"只减不增"拒掉，整局以为自己只剩 2 血）
    def feed(last, reads, **kw):
        rej = (0, 0)
        for r in reads:
            last, rej = filter_hp(last, r, rej, **kw)
        return last
    for got, want, why in (
            (feed(50, [2, 2, 2, 2], rounds_elapsed=1), 50, "一回合掉 48 血，连读再多次也不认"),
            (feed(50, [20, 20, 20], rounds_elapsed=2), 20, "隔了 2 回合掉 30，连读 3 次认"),
            (feed(2, [52, 52, 52, 52], before=52), 52, "误认的 2 在连读 4 次 52 后纠正回来"),
            (feed(2, [52, 52, 52], before=52), 2, "只读到 3 次 52 还不纠正"),
            (feed(40, [60, 60, 60, 60], before=45), 40, "往上纠正不能超过本局认定过的最高血量")):
        ok = got == want
        print(f"[{'OK' if ok else 'FAIL'}] {why} => {got}（期望 {want}）")
        if not ok:
            bad.append(f"血量判据 {why}: 给出 {got}，期望 {want}")
    got, rej = filter_hp(60, 15, (0, 0), first=True, round_no=5)
    mid, rej = filter_hp(got, 57, rej, first=True, round_no=5)
    got, _ = filter_hp(mid, 57, rej, first=True, round_no=5)
    ok = mid != 15 and got == 57
    print(f"[{'OK' if ok else 'FAIL'}] 首读 15 只出现一帧、之后连读 57 => 中间 {mid}、最后 {got}（期望 不是15 / 57）")
    if not ok:
        bad.append(f"首读单帧误读被采信: {got}")
    # 本局第一次读数没有基准可比（中途接手/离线帧），只要合法就该采信
    # 首读也要连续两帧一致才采信（第十三局：重启后首读被遮挡读成 15、真值 57）
    got, rej = filter_hp(60, 10, (0, 0), first=True)
    got, _ = filter_hp(got, 10, rej, first=True)
    ok = got == 10
    print(f"[{'OK' if ok else 'FAIL'}] 本局首次读数 60->10 => {got}（期望 10，别卡在满血）")
    if not ok:
        bad.append(f'本局首次读数没被采信: {got}')
    # **首读也要讲得通**：第 2 回合掉 30 血不可能（2026-09-23 实战被这样骗过一次）。
    for rnd, hp, want, why in ((2, 30, 60, "第2回合掉30血讲不通"),
                               (5, 10, 10, "第5回合剩10血讲得通"),
                               (0, 10, 10, "回合未知(中途接手)不设限")):
        got, rej = filter_hp(60, hp, (0, 0), first=True, round_no=rnd)
        got, _ = filter_hp(got, hp, rej, first=True, round_no=rnd)
        ok = got == want
        print(f"[{'OK' if ok else 'FAIL'}] 首读 回合{rnd} 血{hp} => {got}（{why}）")
        if not ok:
            bad.append(f"首读判据 {why}: 给出 {got}，期望 {want}")
    # 真掉了大血也不能被永久冻住：同一个超限值连读几次就该认
    last, reject = 57, (0, 0)
    for _ in range(3):
        last, reject = filter_hp(last, 30, reject)
    ok = last == 30
    print(f"[{'OK' if ok else 'FAIL'}] 同一超限值连读三次 => {last}（期望 30，别冻住）")
    if not ok:
        bad.append(f"超限值连读三次没被采信：{last}")
    return bad


def check_player_hp(ctx) -> list[str]:
    """六人血条 + 认出哪一行是自己。

    2026-09-22 之前 ROI 整体偏左 50px，罩住的是头像不是数字，
    读出 6/6/5/5/5/15 这种鬼值（真值 52/35/10）；而 `state.hp` 干脆从没被写过。
    我方行靠**昵称是绿色的**认出来（实测绿像素 306 vs 其余 ≤4）。
    """
    from agent.perception.extractors import StateExtractor, _imread
    import cv2
    import numpy as np
    frame = PROJECT_DIR / "data/logs/live_1790074508/phase_prepare_1790074666.png"
    if not frame.exists():
        print("[跳过] 血条 fixture 不存在")
        return []
    ex = StateExtractor(PROJECT_DIR / "assets" / "image")
    img = cv2.cvtColor(_imread(frame), cv2.COLOR_BGR2RGB)
    # **跑两帧**：血量从初始 60 一下掉到 10 属于"大跌"，按规则要连续两帧一致才采信
    # （单帧误读会漏网 —— 实测出现过 57→4，既"变小"又不是"首位被吃"）。
    ex.extract(ctx, img)
    ex._region_hash.clear()          # 模拟"新的一帧、同样的读数"（同一批像素走缓存，按设计不算复核）
    st = ex.extract(ctx, img)
    got = [p.hp for p in st.players]
    want = [52, 35, 10, -1, -1, -1]
    mine = next((p.index for p in st.players if p.name == "我"), None)
    ok = got == want and mine == 2 and st.hp == 10
    print(f"[{'OK' if ok else 'FAIL'}] 六人血量 {got}（期望 {want}）"
          f" 我方行={mine}（期望 2） hp={st.hp}（期望 10）")
    return [] if ok else [f"六人血条读数不对: {got} 我方行={mine} hp={st.hp}"]


def check_equip_popup(ctx) -> list[str]:
    """装备四选一：**按位置选，不读文字**（2026-09-23 定的做法）。

    守三件事：
      1. 次序：弹窗 4 个图标**从左到右 = equip_recipes 的次序**。
         两个独立样本、两件不同的基础装备，都拿图标旁的效果概括对过 cards.json：
           力量腰带：霸者重铠 / 不死鸟之眼 / 魔女斗篷 / 红莲炎甲
           雷鸣刃  ：冥王 / 名刀·司命 / 红魔 / 魔律之刃
         （我一开始按"左列上下、右列上下"猜，把 0 和 1 弄反了 —— 这条就是防它的。）
      2. 漏认一个图标时按左右对称补，补出来的点要落在图标上（图标半径 ~23）。
      3. **反例**：这三张以前都被旧判据当成了弹窗（它数的是随便什么圆，
         实际检出的是等级徽章），现在必须判"不在"。
    """
    from agent.perception.extractors import (_imread, find_equip_slots, find_equip_thumb,
                                             pick_equip_index)
    import cv2
    import numpy as np
    import json
    recipes = json.loads((PROJECT_DIR / "data/equip_recipes.json")
                         .read_text(encoding="utf-8"))["recipes"]
    cases = [
        # (帧, 目标格, 基础装备, 从左到右每个图标的真实位置)
        ("data/logs/fixtures/equip_popup_4options.png", (499, 500), "力量腰带",
         [(385, 458), (433, 370), (539, 370), (589, 458)]),
        ("data/logs/fixtures/equip_popup_leimingren.png", (512, 415), "雷鸣刃",
         [(409, 459), (458, 371), (565, 371), (616, 459)]),
        # 实战抓到的第三件（2026-09-23）。**recipes 原来录错了次序**（错开两位），
        # 想选无尽巨剑点成了纯净苍穹；已按这张图改正。
        ("data/logs/fixtures/equip_popup_fengbaojujian.png", (300, 569), "风暴巨剑",
         [(176, 519), (226, 431), (332, 431), (382, 519)]),
        # 第四件（同一局）。recipes 原来最后两个录反了，想选金色圣剑点成了自然法杖。
        ("data/logs/fixtures/equip_popup_dabang.png", (300, 569), "大棒",
         [(176, 517), (224, 429), (334, 431), (382, 519)]),
        # 第五件：recipes 里原本也录错了次序，是按"id 升序"规律**推出来**再实战核对的，4/4 对上。
        ("data/logs/fixtures/equip_popup_sujizhiqiang.png", (617, 372), "速击之枪",
         [(509, 322), (557, 232), (665, 234), (715, 320)]),
        ("data/logs/fixtures/equip_popup_posuishengbei.png", (617, 372), "破碎圣杯",
         [(511, 320), (559, 232), (663, 234), (715, 322)]),
        # 这张的 👍（炽热支配者上）是半透明的，我肉眼又漏看了一次，模板 0.76 先认出来
        ("data/logs/fixtures/equip_popup_shengzhefadian.png", (529, 372), "圣者法典",
         [(399, 314), (447, 224), (555, 224), (605, 312)]),
        # 第八件，至此 8 件基础装备全部实测核对过，次序全部 = 卡牌 id 升序
        ("data/logs/fixtures/equip_popup_shouhuzhe.png", (672, 430), "守护者之铠",
         [(570, 400), (612, 310), (722, 308), (770, 396)]),
    ]
    # 游戏的 👍 推荐标记：风暴巨剑那张挂在第 2 个（纯净苍穹）上，另外两张没有
    # 大棒那张的 👍（虚无法杖上）是我标注时漏看的，模板先找出来、人工复核确认
    thumbs = {"力量腰带": None, "雷鸣刃": None, "风暴巨剑": 2, "大棒": 0, "速击之枪": 0, "破碎圣杯": 3, "圣者法典": 1, "守护者之铠": 1}
    negatives = [
        ("data/logs/fixtures/equip_not_popup_lifted.png", (300, 569)),
        ("data/logs/fixtures/equip_not_popup_dragging.png", (617, 496)),
        ("data/logs/fixtures/equip_not_popup_prepare.png", (300, 569)),
    ]
    bad = []
    for rel, center, base, truth in cases:
        case_start = len(bad)
        path = PROJECT_DIR / rel
        if not path.exists():
            bad.append(f"装备四选一 fixture 缺失: {rel}")
            continue
        img = cv2.cvtColor(_imread(path), cv2.COLOR_BGR2RGB)
        got = find_equip_slots(img, center)
        for i, (want, have) in enumerate(zip(truth, got)):
            if have is None or abs(have[0] - want[0]) > 12 or abs(have[1] - want[1]) > 12:
                bad.append(f"{base} 第{i}个（{recipes[base][i]}）应在 {want}，认成 {have}")
        # 漏认任意一个都要能补回来
        for k, (x, y) in enumerate(truth):
            # 用**周围一圈的底色**把图标抹平。别用纯色实心圆去"盖"：
            # 亮色实心圆本身就会被认成一个新图标，测出来的"补点成功"是假的。
            ring = np.zeros(img.shape[:2], np.uint8)
            cv2.circle(ring, (x, y), 40, 255, -1)
            cv2.circle(ring, (x, y), 32, 0, -1)
            im = img.copy()
            cv2.circle(im, (x, y), 32, tuple(float(v) for v in img[ring > 0].mean(axis=0)), -1)
            s = find_equip_slots(im, center)[k]
            if s is None or abs(s[0] - x) > 12 or abs(s[1] - y) > 12:
                bad.append(f"{base} 盖掉第{k}个后补成 {s}，真值 {(x, y)}")
        t = find_equip_thumb(img, got)
        if t != thumbs[base]:
            bad.append(f"{base} 👍 应在 {thumbs[base]}，认成 {t}")
        # 选哪个：下标必须合法、名字必须对得上 recipes
        for role in ("tank", "dps"):
            pick = pick_equip_index(PROJECT_DIR / "data", base, role)
            if pick is None or recipes[base][pick[0]] != pick[1]:
                bad.append(f"{base}/{role} 选择结果不一致: {pick}")
        print(f"[{'OK' if len(bad) == case_start else 'FAIL'}] 装备四选一 {base}: "
              + " ".join(f"{n}{p}" for n, p in zip(recipes[base], got)))
    # 不变量：recipes 次序 = 卡牌 id 升序（4 件实测都成立，2026-09-23）。
    # 手改这张表时最容易把它弄乱，而一乱就是**点错装备、日志还显示选对了**。
    cards = json.loads((PROJECT_DIR / "data/cards.json").read_text(encoding="utf-8"))
    items = cards if isinstance(cards, list) else cards.get("cards", cards)
    ids = {c["name"]: int(c["id"]) for c in (items.values() if isinstance(items, dict) else items)
           if isinstance(c, dict) and c.get("name") and str(c.get("id", "")).isdigit()}
    for base, names in recipes.items():
        if [ids.get(n) for n in names] != sorted(ids.get(n, 0) for n in names):
            bad.append(f"recipes[{base}] 不是按卡牌 id 升序: {names}")
    n_before = len(bad)
    for rel, center in negatives:
        path = PROJECT_DIR / rel
        if not path.exists():
            bad.append(f"装备四选一反例 fixture 缺失: {rel}")
            continue
        img = cv2.cvtColor(_imread(path), cv2.COLOR_BGR2RGB)
        if find_equip_slots(img, center)[0] is not None:
            bad.append(f"反例 {rel} 被认成了弹窗")
    print(f"[{'OK' if len(bad) == n_before else 'FAIL'}] 装备四选一反例 {len(negatives)} 张")
    return bad


def check_badges() -> list[str]:
    """等级徽章：真徽章是**深色**的，不是"灰色"的（2026-09-23 用 249 帧量出来的）。

    旧判据"饱和度<75"把真徽章（sat 79~95）扔了、留下卡面上的深色圆，
    准确率只有 37%。这里守两个具体帧：
      - 一个单位站在 (300,569)，徽章在 (282,460)，身上 1 件装备 —— 必须认出来、件数 1
      （旧判据在这一帧上**什么都认不出**：badges={}、equips={}）
    """
    from agent.perception.extractors import _detect_badges, _imread, detect_equip_counts
    import cv2
    path = PROJECT_DIR / "data/logs/fixtures/badge_one_unit_one_equip.png"
    if not path.exists():
        return [f"徽章 fixture 缺失: {path.name}"]
    img = cv2.cvtColor(_imread(path), cv2.COLOR_BGR2RGB)   # 实战里是 RGB
    badges = _detect_badges(img)
    equips = detect_equip_counts(img)
    bad = []
    if badges.get("300,569") is None:
        bad.append(f"(300,569) 的徽章没认出来: {badges}")
    if len(badges) != 1:
        bad.append(f"应只有 1 个单位，认出 {len(badges)} 个: {badges}")
    if equips.get("300,569") != 1:
        bad.append(f"(300,569) 装备件数应为 1，读成 {equips.get('300,569')}")
    print(f"[{'OK' if not bad else 'FAIL'}] 等级徽章/装备件数: {badges} {equips}")
    return bad


def check_name_resolve() -> list[str]:
    """卡名解析：OCR 认花的名字要能兜回来，**但绝不能把噪声当成卡**。

    2026-09-22 实战：「翩跹舞步·新生」的"翩跹"被读成 解冠/曾跃，difflib 判不像，
    整张卡一局都没用上。这一族有 5 张同前缀的牌，区分本来就靠后缀、而后缀认得对，
    所以加了一道字符覆盖率兜底 + "第一名要明显好过第二名"。
    负样本同样重要：曾经把别的玩家昵称和数字当成商店里的牌。
    """
    from agent.perception.extractors import StateExtractor
    ex = StateExtractor(PROJECT_DIR / "assets" / "image")
    want = {
        "解冠舞步·新生": "翩跹舞步·新生",
        "曾跃舞步·新生": "翩跹舞步·新生",
        "体面离坊": "体面离场",
        "孙尚香": "孙尚香",
        "画个圈圈嗨": None,      # 负样本：别的玩家昵称
        "35": None,             # 负样本：数字
        # 唯一前缀兜底（2026-09-23 加）：尾巴被认花时要能救回来……
        "破碎圣书A": "破碎圣杯",
        # ……但**不能把一整句话当卡名**。这条是实战当场打脸的：
        # 「姜子牙），鲁班七号」只是恰好以卡名开头，被兜底认成了「姜子牙」。
        "姜子牙），鲁班七号": None,
        "姜子牙鲁班七号": None,
        # 同前缀有 5 张的一族，光看前缀本来就分不出来，硬猜只会猜错
        "翩跹舞步": None,
    }
    bad = []
    for text, expect in want.items():
        got = ex.resolve_name(text, keep_raw=False)
        if got != expect:
            bad.append(f"卡名解析 {text!r} -> {got!r}（期望 {expect!r}）")
    # **手牌区**只在"会进手牌的牌"里认（2026-09-24：「弈」读成「奕/亦/变」时，全库匹配会落到被动天赋「逐星」，
    # 弈星 于是一直不被当成英雄）。会进手牌的天赋牌（碰碰运气/体面离场/终极复印机）要照常认得出来。
    hand_want = {"奕星": "弈星", "亦星": "弈星", "变星": "弈星", "碰碰运气": "碰碰运气",
                 "体面离场": "体面离场", "终极复印机": "终极复印机", "孙悟": "孙悟空"}
    for text, expect in hand_want.items():
        got = ex._resolve_hand_fragment(text)
        if got != expect:
            bad.append(f"手牌卡名 {text!r} -> {got!r}（期望 {expect!r}）")
    print(f"[{'OK' if not bad else 'FAIL'}] 卡名解析 {len(want)} 例 + 手牌 {len(hand_want)} 例"
          f"（含 5 个负样本）{'' if not bad else bad}")
    return bad


def check_hand_and_pop(ctx) -> list[str]:
    """手牌张数 + 人口。两处都在 2026-09-22 实战里给出过假数据。

    手牌：ROI 的 x 从 100 开始，把左下角的人口数字当成一张牌读进来；
    y 也没贴住卡名行 —— 满手牌时只认出 9/11 张，日志报"手牌:4"而画面已经溢出到
    "开战后删除"。
    人口：ROI 本身是对的（棋盘地面上的 "N/M👤"），但**上限只防变小、不防变大**，
    一次误读出 9 就永久粘住（日志一路 "人口 3/9"，真实上限最多 7）。
    """
    from agent.perception.extractors import StateExtractor, _imread
    import cv2
    cases = [
        ("data/logs/live_1790068521/phase_prepare_1790068762.png", 2, None),
        ("data/logs/live_1790074508/phase_prepare_1790074666.png", 5, (6, 7)),
        # **卡名那一行的高度不固定**：同一局里出现过 y≈653/665/701 三种位置。
        # 这一帧的卡名在 y≈653，而 ROI 一度按另一帧收成 y683~721 ——
        # 结果整条 ROI 一个字都读不到、手牌报 0 张，
        # 买回来的英雄全上不去（2026-09-22 实测：好几个英雄买了没上场）。
        ("data/logs/fixtures/hand_high_row.png", 2, (2, 5)),
    ]
    bad = []
    for rel, want_hand, want_pop in cases:
        path = PROJECT_DIR / rel
        if not path.exists():
            print(f"[跳过] {rel} 不存在")
            continue
        ex = StateExtractor(PROJECT_DIR / "assets" / "image")
        st = ex.extract(ctx, cv2.cvtColor(_imread(path), cv2.COLOR_BGR2RGB))
        got_pop = (st.population, st.population_cap) if st.population_valid else None
        ok = len(st.hand_slots) == want_hand and got_pop == want_pop
        print(f"[{'OK' if ok else 'FAIL'}] {Path(rel).name}: 手牌 {len(st.hand_slots)}"
              f"（期望 {want_hand}）人口 {got_pop}（期望 {want_pop}）")
        if not ok:
            bad.append(f"{rel}: 手牌 {len(st.hand_slots)}!={want_hand} 或 人口 {got_pop}!={want_pop}")
    return bad


def check_hand_awakened(ctx) -> list[str]:
    """**挨着的同名牌**必须一张不少（2026-09-23 截图）。

    这一帧手上：枫叶舞 / 风暴巨剑 / 弈星 / 弈星 / 东皇太一 / 干将莫邪 / 干将莫邪(觉醒)。
    旧读法只认出 5 张 —— 同名去重阈值 120 比真实牌距（~100）还大，
    **两张挨着的弈星、两张挨着的干将莫邪各被吞掉一张**，吞掉的正好是觉醒卡，
    于是它一整回合"不存在"，决策层拿普通卡往旧格子合成、失败、搁置，主C 没重新上场。
    """
    from agent.perception.extractors import StateExtractor, _imread
    import cv2
    path = PROJECT_DIR / "data/logs/fixtures/hand_awakened_dupes.png"
    if not path.exists():
        return [f"fixture 缺失: {path.name}"]
    want = ["枫叶舞", "风暴巨剑", "弈星", "弈星", "东皇太一", "干将莫邪", "干将莫邪"]
    ex = StateExtractor(PROJECT_DIR / "assets" / "image")
    st = ex.extract(ctx, cv2.cvtColor(_imread(path), cv2.COLOR_BGR2RGB))
    got = [sl["name"] for sl in st.hand_slots]
    awake = [sl["name"] for sl in st.hand_slots if sl.get("awakened")]
    bad = []
    if got != want:
        bad.append(f"手牌 {got} != {want}")
    # 2026-09-25 起**感知层用手牌觉醒模型直接标**（概率 ≥0.99；测试集 11/0/0），不再等点开确认。
    # 这张 fixture 只有最右那张干将莫邪是觉醒卡 —— 守"标对、且只标它一张"。
    right = max((sl for sl in st.hand_slots if sl["name"] == "干将莫邪"), key=lambda s: s["x"], default=None)
    marked = [sl for sl in st.hand_slots if sl.get("awakened")]
    if not (len(marked) == 1 and marked[0] is right):
        bad.append(f"觉醒卡应只标最右那张干将莫邪，实际标了 {[(m['name'], m['x']) for m in marked]}")
    print(f"[{'OK' if not bad else 'FAIL'}] 挨着的同名牌+觉醒卡: {got} 觉醒={awake}")
    return bad


def check_inspect(ctx) -> list[str]:
    """点开手牌看放大详情 + 漏读推断 + 拖牌前抓取点校验（live_match_agent），用实战真帧。

    第一版的检测坐标是按一张外部截图换算的（那张不是 16:9），「说明」被切掉 ->
    明明打开了却判"没打开"，还因此**没去关**。这里守：能认出打开了、读得出卡名、
    关的时候点的是场馆背景（不是左侧玩家列表）、关完确认关掉了、同一张牌不重复点。
    """
    import cv2
    import live_match_agent as L
    from agent.decision.rules import RuleEngine
    from agent.perception.extractors import StateExtractor, _imread
    from agent.perception.state import GameState
    opened = PROJECT_DIR / "data/logs/fixtures/inspect_big_card_normal.png"
    closed = PROJECT_DIR / "data/logs/fixtures/hand_awakened_dupes.png"
    if not opened.exists() or not closed.exists():
        return ["看牌 fixture 缺失"]
    f_open = cv2.cvtColor(_imread(opened), cv2.COLOR_BGR2RGB)
    f_closed = cv2.cvtColor(_imread(closed), cv2.COLOR_BGR2RGB)
    shots = iter([f_open] + [f_closed] * 50)

    class FakeExec:
        taps = []

        def _tap(self, x, y):
            self.taps.append((x, y))

    ex = StateExtractor(PROJECT_DIR / "assets" / "image")
    rules = RuleEngine(PROJECT_DIR / "data" / "chess_pieces.json",
                       cards_db_path=PROJECT_DIR / "data" / "cards.json")
    rules._field = {"孙膑": (1027, 572)}
    st = GameState(phase="prepare", round_no=6)
    st.hand_slots = [{"name": "孙膑", "x": 797, "y": 663, "name_y": 688}]
    orig_cap, orig_sleep = L.screencap, L.time.sleep
    L.screencap = lambda _c: next(shots)
    L.time.sleep = lambda _s: None
    L._INSPECT.update({"flags": {}, "round": -1, "count": 0, "gaps": set(), "alias": {}})
    fx = FakeExec()
    bad = []
    try:
        L.inspect_hand(ctx, None, fx, ex, rules, st)
        first = list(fx.taps)
        L.inspect_hand(ctx, None, fx, ex, rules, st)          # 同一回合同一张：不该再点
        # 抓牌校验：真帧上 东皇太一 在 x=798
        L.screencap = lambda _c: f_closed
        act_ok = {"type": "drag_card", "piece": "东皇太一", "src": (798, 663), "dst": (529, 372)}
        act_moved = {"type": "drag_card", "piece": "东皇太一", "src": (683, 663), "dst": (529, 372)}
        act_gone = {"type": "drag_card", "piece": "李元芳", "src": (683, 663), "dst": (529, 372)}
        r_ok = L.confirm_hand_src(ctx, None, ex, act_ok)
        r_moved = L.confirm_hand_src(ctx, None, ex, act_moved)
        r_gone = L.confirm_hand_src(ctx, None, ex, act_gone)
    finally:
        L.screencap, L.time.sleep = orig_cap, orig_sleep
    flag = L._INSPECT["flags"].get(("孙膑", 1))
    import numpy as _np
    from agent.perception.extractors import _hand_awaken_by_model
    model_on = _hand_awaken_by_model(_np.zeros((720, 1280, 3), _np.uint8), []) is not None
    if st.hand_slots[0].get("awakened"):
        bad.append("普通卡被标成了觉醒卡")
    if model_on:
        # 2026-09-25 起：觉醒与否交给手牌觉醒模型，"场上也有同名"不再触发点开（避免重复打开查看手牌）
        if first:
            bad.append(f"有手牌觉醒模型时不该为'场上也有同名'点开看，实际点了 {first}")
    else:
        if flag is not False:
            bad.append(f"孙膑放大图应认成普通卡（False），结果 {flag}")
        if len(first) != 2 or first[1] != L.TAP_CLOSE_BIG_CARD:
            bad.append(f"应该是 点卡 + 点场馆背景关掉 两下，实际 {first}")
    if any(x < 230 and 60 < y < 430 for x, y in fx.taps):
        bad.append(f"点到了左侧玩家列表区域: {fx.taps}")
    if len(fx.taps) != len(first):
        bad.append(f"同一张牌同一回合又点了一次: {fx.taps}")
    if not r_ok or tuple(act_ok["src"]) != (798, 663):
        bad.append(f"抓取点本来就对，却没放行或改了位置: {r_ok} {act_ok['src']}")
    if not r_moved or abs(act_moved["src"][0] - 798) > 10:
        bad.append(f"手牌挪了位置没重新定位: {r_moved} {act_moved['src']}")
    if r_gone:
        bad.append("手里没有的牌也放行了")
    # 放大图"开没开"：效果牌的放大图**没有右侧「说明」栏**（实战误判过），要靠卡名认
    f_effect = cv2.cvtColor(_imread(PROJECT_DIR / "data/logs/fixtures/inspect_big_card_effect.png"),
                            cv2.COLOR_BGR2RGB)
    f_awake = cv2.cvtColor(_imread(PROJECT_DIR / "data/logs/fixtures/inspect_big_card_awake.png"),
                           cv2.COLOR_BGR2RGB)
    if not L._big_card_open(ctx, f_effect, ex):
        bad.append("效果牌（1阶效果锦囊）的放大图没认出来")
    if L._big_card_open(ctx, f_closed, ex):
        bad.append("普通手牌画面被当成放大图开着")
    top = [t for t, _b in L._ocr_boxes(ctx, f_awake, L.ROI_BIG_CARD_TOP)]
    nm = [ex.resolve_name(t) for t, _b in L._ocr_boxes(ctx, f_awake, L.ROI_BIG_CARD_NAME)]
    if not any(L.AWAKEN_RE.search(t) for t in top) or "曜" not in nm:
        bad.append(f"觉醒曜的放大图：顶部 {top} 卡名 {nm}（期望读出 AWAKENED 和 曜）")
    # 漏读推断（纯几何）：684 -> 928 隔了两个牌距，中间少一张
    gaps = L._hand_gaps([{"x": x} for x in (267, 438, 562, 684, 928, 1050)])
    if len(gaps) != 1 or abs(gaps[0] - 806) > 15:
        bad.append(f"漏读推断不对: {gaps}（期望 [~806]）")
    if L._hand_gaps([{"x": x} for x in (267, 440, 562, 683, 798, 929, 1051)]):
        bad.append("手牌一张不少，却推断出了漏读")
    print(f"[{'OK' if not bad else 'FAIL'}] 看牌流程: 点击 {fx.taps} 结论 {flag} | "
          f"抓牌校验 {r_ok}/{r_moved}->{act_moved['src']}/{r_gone} | 漏读推断 {gaps}")
    return bad


def check_board_inspect(ctx) -> list[str]:
    """点开场上单位（live_match_agent.inspect_board）：英雄详情面板要认得出来、卡名要读得对。"""
    import cv2
    import live_match_agent as L
    from agent.perception.extractors import StateExtractor, _imread
    ex = StateExtractor(PROJECT_DIR / "assets" / "image")
    panel = cv2.cvtColor(_imread(PROJECT_DIR / "data/logs/fixtures/hero_panel_blocks_levelup.png"),
                         cv2.COLOR_BGR2RGB)
    plain = cv2.cvtColor(_imread(PROJECT_DIR / "data/logs/fixtures/hand_awakened_dupes.png"),
                         cv2.COLOR_BGR2RGB)
    bad = []
    if not L._hero_panel_up(ctx, panel):
        bad.append("英雄详情面板没认出来")
    if L._hero_panel_up(ctx, plain):
        bad.append("普通画面被当成英雄详情面板")
    names = [ex.resolve_name(t) for t, _b in L._ocr_boxes(ctx, panel, L.ROI_UNIT_PANEL_NAME)]
    if "孙膑" not in names:
        bad.append(f"详情面板上的卡名读成 {names}（期望 孙膑）")
    print(f"[{'OK' if not bad else 'FAIL'}] 看场：面板 {L._hero_panel_up(ctx, panel)}/"
          f"{L._hero_panel_up(ctx, plain)} 卡名 {names}")
    return bad


def check_unit_anchors() -> list[str]:
    """徽章 -> 棋盘格 的映射必须**只认真实单位**。

    2026-09-22 发现：棋手不可能站在棋盘上，坐标定位有问题。
    根因是映射太松（最近格子 + 曼哈顿容差 70），把场外的东西硬吸到棋盘格上。
    但**不能靠收窄 dy 来区分棋手**：全历史帧统计 dy 连续分布在 80~135
    （不同英雄模型身高差很多），我一度按单个样本收成 126±15，结果一个单位都检不到。
    现在容差放宽，锚点**只用来"加占用"**（避开格子）——
    多认一个是保守方向、无害；**少认才危险**（会让决策以为那格空着）。
    所以这个用例只断言"该有的都在"，不断言"多余的没有"。
    """
    from agent.perception.extractors import _imread, detect_unit_anchors
    import cv2
    frame = PROJECT_DIR / "data/logs/live_1790074508/phase_prepare_1790074666.png"
    if not frame.exists():
        print("[跳过] 锚点 fixture 不存在")
        return []
    got = set(detect_unit_anchors(cv2.cvtColor(_imread(frame), cv2.COLOR_BGR2RGB)))
    # 这一帧人口 6/7，画面上 6 个单位逐个核对过（2026-09-23）：
    #   四个 1 级 + 一个 4 级（深色徽章），外加等级 58 和 24 的两个**蓝色徽章** ——
    #   后两个里写的是两位数，白色笔画把均值拉过了阈值，改用中位亮度之前一直漏认；
    #   (917,570) 那个 1 级单位旧检测器也一直没认出来，原用例里压根没写它。
    must = {"1027,572", "529,372", "718,371", "617,372", "300,569", "917,570"}
    missing = must - got
    ok = not missing
    print(f"[{'OK' if ok else 'FAIL'}] 单位锚点 {sorted(got)}（必须包含 {sorted(must)}）")
    return [] if ok else [f"单位锚点漏检: 缺 {sorted(missing)}"]


def check_vision_units() -> list[str]:
    """D-FINE 视觉检测要把旧徽章检测漏掉的单位格补进 unit_anchors。

    帧 f0061（验证集，人口 7）：人工标注 A3/B3/A4/B4/A5/C1/D1，主 agent 逐个徽章核对过；
    旧徽章检测漏了 A3、C1（C1 是固定盲区：被 D1 挡住、徽章贴着左侧玩家列表）。
    """
    import cv2
    import json as _json
    from agent.perception import vision_units
    from agent.perception.extractors import OWN_BOARD_ROWS, _imread, _vision_fill_anchors, cell_key
    from agent.perception.state import GameState
    if not vision_units.enabled():
        print("[跳过] 视觉单位检测未启用（没有模型或 WZWXQ_VISION=0）")
        return []
    frames = {(_r := _json.loads(l))["id"]: _r for l in (PROJECT_DIR / "data/vision/pool_v1/frames.jsonl").open(encoding="utf-8")}
    frame = PROJECT_DIR / frames["f0061"]["frame"]
    if not frame.exists():
        print("[跳过] 视觉 fixture 不存在")
        return []
    names = {f"{'ABCD'[r]}{i + 1}": cell_key(tuple(c)) for r, row in enumerate(OWN_BOARD_ROWS) for i, c in enumerate(row)}
    want = {names[n] for n in ("A3", "B3", "A4", "B4", "A5", "C1", "D1")}
    st = GameState(phase="prepare")
    _vision_fill_anchors(cv2.cvtColor(_imread(frame), cv2.COLOR_BGR2RGB), st)
    got = set(st.vision_units)
    ok = got == want and names["C1"] in st.unit_anchors
    print(f"[{'OK' if ok else 'FAIL'}] 视觉单位检测 f0061：{sorted(got)}（期望 {sorted(want)}，含 C1）")
    return [] if ok else [f"视觉单位检测不对：多 {sorted(got - want)} 缺 {sorted(want - got)}"]


def check_awaken_progress() -> list[str]:
    """觉醒进度 = 等级徽章外圈亮了几段（左→上→右，2026-09-24 实机确认）。过渡用的像素读法。
    两张证据帧：40 级孙膑 左+上 = 2（300,569）；另一帧 4 级曜 / 56 级甄姬 各亮左段 = 1、3 级阿轲 = 0。"""
    import cv2
    from agent.perception.extractors import _imread, read_awaken_progress
    cases = [("awaken_2of3_sunbin40.png", {"300,569": 2}),
             ("awaken_1of3_board.png", {"617,372": 1, "1027,572": 1, "718,371": 0})]
    bad = []
    for fn, want in cases:
        f = PROJECT_DIR / "data/logs/fixtures" / fn
        if not f.exists():
            print(f"[跳过] 觉醒进度 fixture {fn} 不存在")
            continue
        got = read_awaken_progress(cv2.cvtColor(_imread(f), cv2.COLOR_BGR2RGB))
        wrong = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
        print(f"[{'OK' if not wrong else 'FAIL'}] 觉醒进度 {fn}：{got}（期望包含 {want}）")
        if wrong:
            bad.append(f"觉醒进度读错 {fn}：{wrong}")
    return bad


def check_auction_all_lots() -> list[str]:
    """拍品要读全（2026-10-11 第 566 局：6 件只认出 4 件，单字名 镜 / 瑶 被"至少 2 个汉字"过滤掉，Laya 只能在剩下的里挑）。
    先按卡等距居中推算张数（= 剩余玩家数），缺的位置再单独补读。"""
    import cv2
    from agent.perception.extractors import StateExtractor, _imread
    from agent.perception.state import GameState
    f = PROJECT_DIR / "data/logs/fixtures/auction_6_single_char.png"
    if not f.exists():
        print("[跳过] auction_6_single_char.png 不存在")
        return []
    st = GameState(phase="auction")
    StateExtractor(PROJECT_DIR / "assets" / "image")._extract_auction(None, cv2.cvtColor(_imread(f), cv2.COLOR_BGR2RGB), st)
    got = [l.piece for l in st.auction.lots]
    want = ["少司缘", "姬小满", "镜", "猪八戒", "瑶", "百里守约"]
    ok = got == want and st.auction.expected == 6
    print(f"[{'OK' if ok else 'FAIL'}] 拍品读全：{got}（推算 {st.auction.expected} 张；期望 {want}）")
    return [] if ok else ["拍品没读全"]


def check_auction_bidders() -> list[str]:
    """拍卖出价人列表：最上面一行是领先者，我方昵称（英文）是绿色（2026-09-24 数据局逐秒截图 f15）。
    原来要求名字含汉字，自己那一行被整行丢掉 → 永远看不出被超价。"""
    import cv2
    from pathlib import Path as _P
    from agent.perception.extractors import StateExtractor, _imread
    from agent.perception.state import GameState
    f = PROJECT_DIR / "data/logs/fixtures/auction_bidders_me_leading.png"
    if not f.exists():
        print("[跳过] 拍卖 fixture 不存在")
        return []
    im = cv2.resize(_imread(f), (1280, 720))
    st = GameState(phase="auction")
    StateExtractor(PROJECT_DIR / "assets" / "image")._extract_auction(None, cv2.cvtColor(im, cv2.COLOR_BGR2RGB), st)
    got = {l.piece: (l.top_bidder, l.top_bid) for l in st.auction.lots}
    ok = (got.get("众星拱月") == ("我", 2) and got.get("沧海拾遗", ("",))[0] == "ontcy"
          and st.auction.next_price == 3)
    print(f"[{'OK' if ok else 'FAIL'}] 拍卖出价人：{got}，按钮现价 {st.auction.next_price}"
          f"（期望 众星拱月=我/2，沧海拾遗=ontcy，按钮 3）")
    return [] if ok else [f"拍卖出价人读错：{got}"]


def check_solo_join(ctx) -> list[str]:
    """组队弹窗：「独自战斗」是标题，要点它那一列下面的「加入」（第三轮第 4 局点了两次标题没反应，截图确认）。"""
    import cv2
    import live_match_agent as L
    from agent.perception.extractors import _imread
    f = PROJECT_DIR / "data/logs/fixtures/team_choice_solo_join.png"
    if not f.exists():
        print("[跳过] 组队弹窗 fixture 不存在")
        return []
    hit = L.find_solo_button(ctx, cv2.cvtColor(_imread(f), cv2.COLOR_BGR2RGB))
    ok = bool(hit) and abs(hit[1] - 953) <= 40 and abs(hit[2] - 507) <= 20
    print(f"[{'OK' if ok else 'FAIL'}] 组队弹窗点「加入」：{hit}（期望 ≈(953,507)）")
    return [] if ok else [f"独自战斗的加入按钮定位不对：{hit}"]


def check_battle_away() -> list[str]:
    """去对手棋盘打时右侧多两行「打开商店」，「战斗统计」被挤到 y≈202 —— 原来 36 秒认不出战斗阶段（match14 复盘）。"""
    import cv2
    from agent.perception.extractors import StateExtractor, _imread
    f = PROJECT_DIR / "data/logs/fixtures/battle_away_stat_y202.png"
    if not f.exists():
        print("[跳过] 客场战斗 fixture 不存在")
        return []
    ph = StateExtractor(PROJECT_DIR / "assets" / "image").detect_phase(None, cv2.cvtColor(_imread(f), cv2.COLOR_BGR2RGB))
    ok = ph == "battle"
    print(f"[{'OK' if ok else 'FAIL'}] 客场战斗阶段：{ph}（期望 battle）")
    return [] if ok else [f"客场战斗认成 {ph}"]


def check_level_cost_from_xp(ctx) -> list[str]:
    """升级费按经验条算（2026-09-24 视觉验证局第 6 回合）：画面 3 级、经验 1/17 → 升级费 16。
    当时粘着 2 级时的旧值 7，以为 10 能量够升级，整回合空转。"""
    import cv2
    from agent.perception.extractors import StateExtractor, _imread
    from agent.perception.state import GameState
    frame = PROJECT_DIR / "data/logs/fixtures/xp_1of17_r6.png"
    if not frame.exists():
        print("[跳过] 经验条 fixture 不存在")
        return []
    ex = StateExtractor(PROJECT_DIR / "assets" / "image")
    ex._last_level_cost = 7                       # 模拟上一级粘住的旧值
    st = GameState(phase="prepare", level_cost=7)
    ex._extract_lord_level(cv2.cvtColor(_imread(frame), cv2.COLOR_BGR2RGB), st, context=ctx)
    ex._finalize_level_cost(st)
    ok = (st.xp, st.xp_needed, st.level, st.level_cost) == (1, 17, 3, 16)
    print(f"[{'OK' if ok else 'FAIL'}] 经验条算升级费：经验 {st.xp}/{st.xp_needed} 等级 {st.level} 升级费 {st.level_cost}（期望 1/17、3 级、16）")
    return [] if ok else [f"升级费没按经验条算：{st.xp}/{st.xp_needed} lv{st.level} cost{st.level_cost}"]


def check_hero_panel(ctx) -> list[str]:
    """英雄详情面板要能认出来 —— 它盖住升级按钮，会导致"有钱也升不了级"。

    2026-09-22 实战：攒到 36 能量、升级费只要 8，连着几十帧都在
    "想升级但画面上没找到「升级」按钮" —— 因为左下角那块变成了「出售」「剩余数量」。
    而这时阶段仍判成 prepare，unknown 兜底不会触发，所以必须单独认出来。
    负样本同样重要：正常准备帧不能误报，否则会一直去点空地。
    """
    from agent.perception.extractors import StateExtractor, _imread
    import cv2
    cases = [("data/logs/fixtures/hero_panel_blocks_levelup.png", True),
             ("data/logs/live_1790074508/phase_prepare_1790074666.png", False)]
    bad = []
    for rel, want in cases:
        path = PROJECT_DIR / rel
        if not path.exists():
            print(f"[跳过] {rel} 不存在")
            continue
        ex = StateExtractor(PROJECT_DIR / "assets" / "image")
        st = ex.extract(ctx, cv2.cvtColor(_imread(path), cv2.COLOR_BGR2RGB))
        ok = st.hero_panel_open == want
        print(f"[{'OK' if ok else 'FAIL'}] 英雄详情面板 {Path(rel).name}: "
              f"{st.hero_panel_open}（期望 {want}）")
        if not ok:
            bad.append(f"英雄详情面板判定错误 {rel}: {st.hero_panel_open} != {want}")
    return bad


def check_match_0923(ctx) -> list[str]:
    """2026-09-23 TensorRT 那一局暴露的问题，每条一个证据帧（data/logs/fixtures/）。"""
    from agent.perception import extractors as E
    from agent.perception.extractors import StateExtractor, _imread
    from agent.perception.state import GameState
    import cv2
    F = PROJECT_DIR / "data" / "logs" / "fixtures"
    load = lambda n: cv2.cvtColor(_imread(F / n), cv2.COLOR_BGR2RGB)   # noqa: E731
    bad = []

    def expect(ok, what):
        print(f"[{'OK' if ok else 'FAIL'}] {what}")
        if not ok:
            bad.append(what)

    # 1. 能量：闪电图标别认成数字（7 曾读成 37）
    ex = StateExtractor(PROJECT_DIR / "assets" / "image")
    st = ex.extract(ctx, load("energy_icon_7.png"))
    expect(st.gold == 7, f"能量白字读法 energy_icon_7: {st.gold}（期望 7）")
    # 2. 满级 + 天赋未领取（第 15 回合挂了一整回合）
    ex = StateExtractor(PROJECT_DIR / "assets" / "image")
    st = ex.extract(ctx, load("talent_pending_maxed.png"))
    expect(st.lord_maxed and st.level == 6 and (st.level_cost or -1) < 0,
           f"满级识别: maxed={st.lord_maxed} level={st.level} cost={st.level_cost}（期望 True/6/-1）")
    expect(bool(st.talent_pending), f"「天赋未领取」: {st.talent_pending}（期望有坐标）")
    expect(st.gold == 36, f"能量 talent_pending_maxed: {st.gold}（期望 36）")
    # 3. 天赋界面靠卡面「天赋」标签也认得出（满级时底部那行字没有）
    ph = StateExtractor(PROJECT_DIR / "assets" / "image").detect_phase(ctx, load("talent_tags.png"))
    expect(ph == "talent", f"天赋界面 talent_tags: {ph}（期望 talent）")
    # 3b. 选完天赋弹出的**放大详情**（也带一个「天赋」标签）不是三选一；真三选一照样认得出
    ph = StateExtractor(PROJECT_DIR / "assets" / "image").detect_phase(ctx, load("talent_enlarged_detail.png"))
    expect(ph != "talent", f"天赋放大详情 talent_enlarged_detail: {ph}（期望不是 talent）")
    ph = StateExtractor(PROJECT_DIR / "assets" / "image").detect_phase(ctx, load("talent_real_0923.png"))
    expect(ph == "talent", f"天赋三选一 talent_real_0923: {ph}（期望 talent）")
    # 3c. 满级的橙金色天赋卡：「天赋」读成「天膜」「天娱」、TALENT 读成 LENT —— 也要认得出
    ph = StateExtractor(PROJECT_DIR / "assets" / "image").detect_phase(ctx, load("talent_maxed_open.png"))
    expect(ph == "talent", f"满级天赋三选一 talent_maxed_open: {ph}（期望 talent）")
    # 3c'. 天赋卡名要**直接从卡名区读出来**，不能全靠描述反查兜底（2026-09-24：原 ROI (360,50) 从卡名中心开始切，
    #      卡名只认出 31%、金底卡整个在框外；描述区又切掉了第一行）。两张图（普通 / 满级金底）× 3 张卡。
    from agent.perception.extractors import _ocr as _ocr1, ROI_TALENT_NAME_Y as _TNY, TALENT_CARD_X as _TCX
    im = load("talent_real_0923.png")
    direct = [ex.resolve_name(_ocr1(ctx, im, (x0 + 20, _TNY[0], 200, _TNY[1])), keep_raw=False) for x0 in _TCX]
    expect(direct == ["巨像", "资深猎头", "虚妄转生"], f"天赋卡名直读 talent_real_0923: {direct}（期望 巨像/资深猎头/虚妄转生）")
    # 满级金底卡：阿离满级专属天赋**卡牌库里没收录**，只查原文读得对不对（旧 ROI 读成"洛央波取""融欧馈赔"）
    im = load("talent_maxed_open.png")
    raw = [_ocr1(ctx, im, (x0 + 20, _TNY[0], 200, _TNY[1])) for x0 in _TCX]
    expect(all(w in r for w, r in zip(("落英汲取", "枫之洪流", "翩跹馈赠"), raw)),
           f"满级天赋卡名原文 talent_maxed_open: {raw}（期望含 落英汲取/枫之洪流/翩跹馈赠）")
    # 3c''. 2026-09-24 游戏更新：「战斗统计」被新提示栏挤到 y≈150，战斗阶段整局认不出
    ph = StateExtractor(PROJECT_DIR / "assets" / "image").detect_phase(ctx, load("battle_newui_0924.png"))
    expect(ph == "battle", f"新界面战斗 battle_newui_0924: {ph}（期望 battle）")
    # 人口数字颜色：白=满员（旁边等级徽章挡住"6"的左半边，读数会是 5/6）；红=拖着新英雄、放下就超（实机截图）
    from agent.perception.extractors import pop_color as _pc
    st = StateExtractor(PROJECT_DIR / "assets" / "image").extract(ctx, load("pop_white_full_0924.png"))  # 新识别器：人口上限只增不减，不能带着前面帧的 7
    expect(st.population == 6 and st.population_cap == 6 and st.population_valid,
           f"满员白色人口 pop_white_full_0924: {st.population}/{st.population_cap} valid={st.population_valid}（期望 6/6）")
    expect(_pc(load("pop_red_drag_user_0924.png")) == "red", "拖动时人口变红 pop_red_drag_user_0924（期望 red）")
    expect(_pc(load("pop_green_5of6_0924.png")) is None, "没满的绿色人口不该判成 white/red")
    # 3d. 拍卖顶部横幅被遮暗、读不出字：靠右下角「出价」按钮认出拍卖，不能被当成 N 选 1
    ph = StateExtractor(PROJECT_DIR / "assets" / "image").detect_phase(ctx, load("auction_dim_banner.png"))
    expect(ph == "auction", f"横幅被遮暗的拍卖 auction_dim_banner: {ph}（期望 auction）")
    # 4. 「任务达成」动画卡（写着"强化牌"）不是棋手特性三选一
    ph = StateExtractor(PROJECT_DIR / "assets" / "image").detect_phase(ctx, load("lord_boost_fake_task_done.png"))
    expect(ph != "lord_boost", f"任务达成动画 lord_boost_fake_task_done: {ph}（期望不是 lord_boost）")
    # 5. 右侧已获得天赋列不是 N 选 1
    ex = StateExtractor(PROJECT_DIR / "assets" / "image")
    ph = ex.detect_phase(ctx, load("card_pick_fake_talent_col.png"))
    probe = GameState()
    ex._detect_card_pick(ctx, load("card_pick_fake_talent_col.png"), probe)
    # 这一帧是回合最后 2 秒、「准备」按钮已经没了，判成 unknown 是对的；要紧的是**不能是 card_pick**
    expect(ph != "card_pick" and not probe.card_pick,
           f"天赋列误认 card_pick_fake_talent_col: phase={ph} card_pick={probe.card_pick}（期望不是 card_pick）")
    # 5b. 商店的牌右上角有价格角标，N 选 1 没有（2026-09-23）—— 开着的商店不能认成 N 选 1
    probe = GameState()
    StateExtractor(PROJECT_DIR / "assets" / "image")._detect_card_pick(ctx, load("shop_open_x_button.png"), probe)
    expect(not probe.card_pick, f"开着的商店 shop_open_x_button: card_pick={probe.card_pick}（期望 False）")
    probe = GameState()
    StateExtractor(PROJECT_DIR / "assets" / "image")._detect_card_pick(ctx, load("card_pick_3choices.png"), probe)
    expect(probe.card_pick and len(probe.card_options) == 3,
           f"三选一 card_pick_3choices: {[o['name'] for o in probe.card_options]}（期望 3 张）")
    # 5b'. 英雄 N 选 1：卡面右上角落在橙色场馆/红色立绘上，"偏红"但**没有价格数字** —— 不是商店
    #      （2026-09-23 第七局 5阶英雄兑换券 弹出的三选一就是这么一直没选）
    for fx, want in (("card_pick_heroes_3.png", {"明世隐", "赵云", "貂蝉"}),
                     ("card_pick_heroes_red.png", {"钟馗", "马超"})):
        probe = GameState()
        StateExtractor(PROJECT_DIR / "assets" / "image")._detect_card_pick(ctx, load(fx), probe)
        got = {o["name"] for o in probe.card_options}
        expect(probe.card_pick and want <= got, f"英雄N选1 {fx}: {sorted(got)}（期望含 {sorted(want)}）")
    probe = GameState()
    StateExtractor(PROJECT_DIR / "assets" / "image")._detect_card_pick(
        ctx, load("card_pick_fake_battle_tips.png"), probe)
    probe = GameState()
    StateExtractor(PROJECT_DIR / "assets" / "image")._detect_card_pick(
        ctx, load("card_pick_fake_unit_detail.png"), probe)
    expect(not probe.card_pick, f"战斗中的单位详情+飘字 card_pick_fake_unit_detail: "
                                f"{[o['name'] for o in probe.card_options]}（期望不是 N 选 1）")
    # 2026-10-15 第 649 局：二选一（孙悟空 / 太乙真人）孙悟空的卡名没读出、描述里的「闪现」被认成「闪现突击者」(y=265) → 卡 3 分钟
    probe = GameState()
    StateExtractor(PROJECT_DIR / "assets" / "image")._detect_card_pick(
        ctx, load("card_pick_2choices_desc_misread.png"), probe)
    got = [o["name"] for o in probe.card_options]
    expect(probe.card_pick and got == ["孙悟空", "太乙真人"], f"二选一（描述误读）card_pick_2choices_desc_misread: {got}（期望 孙悟空/太乙真人）")
    # 5c'. 觉醒卡按颜色认候选（冰蓝、几乎无深色）：干将莫邪（最右）/ 孙膑；普通卡不能进候选
    from agent.perception.extractors import awaken_tint_spans
    sp = awaken_tint_spans(load("hand_awakened_dupes.png"))
    expect(len(sp) == 1 and sp[0][0] > 950, f"觉醒色 hand_awakened_dupes: {sp}（期望只有最右那张 干将莫邪）")
    sp = awaken_tint_spans(load("hand_awakened_sunbin.png"))
    expect(len(sp) == 1 and sp[0][0] <= 792 <= sp[0][1], f"觉醒色 hand_awakened_sunbin: {sp}（期望只有 孙膑 那张）")
    sp = awaken_tint_spans(load("hand_awakened_gold.png"))
    expect(len(sp) == 1 and sp[0][0] <= 721 <= sp[0][1], f"金色觉醒 hand_awakened_gold: {sp}（期望只有 孙膑 那张）")
    # 拖拽回执用新建的 GameState 当探针读人口（phase=unknown）——必须读得出来，
    # 否则"人口 +1 = 上场成功"整条判据失效（第十一局：只允许 prepare 时，上场成功全被判失败）
    probe = GameState()
    StateExtractor(PROJECT_DIR / "assets" / "image")._extract_population(ctx, load("hand_awakened_sunbin.png"), probe)
    expect(probe.population_valid and (probe.population, probe.population_cap) == (5, 6),
           f"回执探针读人口 hand_awakened_sunbin: {probe.population}/{probe.population_cap} valid={probe.population_valid}（期望 5/6 有效）")
    probe = GameState(phase="battle")
    StateExtractor(PROJECT_DIR / "assets" / "image")._extract_population(ctx, load("hand_awakened_sunbin.png"), probe)
    expect(not probe.population_valid, "战斗画面的人口必须判无效")
    # 棋手等级按经验条 "N/M" 读（M=升级所需经验，7/13/17/23/27 对应 1~5 级），比点数徽章可靠
    for fx, want in (("lord_xp_0_23.png", 4), ("hand_awakened_sunbin.png", 3), ("hand_awakened_gold.png", 2),
                     ("talent_pending_maxed.png", 6)):
        st = GameState(phase="prepare")
        StateExtractor(PROJECT_DIR / "assets" / "image")._extract_lord_level(load(fx), st, ctx)
        expect(st.level == want, f"棋手等级（经验条）{fx}: {st.level}（期望 {want}）")
    sp = awaken_tint_spans(load("hand_stacked_9.png"))
    expect(not sp, f"觉醒色 hand_stacked_9（9 张普通牌）: {sp}（期望空）")
    expect(not probe.card_pick, f"战斗中飘的天赋说明 card_pick_fake_battle_tips: "
                                f"{[o['name'] for o in probe.card_options]}（期望不是 N 选 1）")
    # 5c. 免费刷新次数「×N」要读得出来（原来整局读成 0，一次没用过）
    st = GameState(phase="prepare", shop_open=True)
    StateExtractor(PROJECT_DIR / "assets" / "image")._extract_economy(ctx, load("shop_open_x_button.png"), st)
    expect(st.free_refresh == 5, f"免费刷新次数 shop_open_x_button: {st.free_refresh}（期望 5）")
    # 6. 拍到之后（「已获得拍品」）不再出价
    ex = StateExtractor(PROJECT_DIR / "assets" / "image")
    st = ex.extract(ctx, load("auction_won.png"))
    expect(st.phase == "auction" and st.auction is not None and st.auction.won,
           f"拍卖已获得 auction_won: phase={st.phase} won={getattr(st.auction, 'won', None)}（期望 auction/True）")
    # 6b. 「等待拍卖中」（拍卖还没开）：右侧天赋列不能被当成拍品（2026-09-26 第四轮第 4 局，点出了天赋说明浮窗）
    st = ex.extract(ctx, load("auction_waiting.png"))
    lots = [getattr(l, "piece", None) for l in ((st.auction.lots if st.auction else None) or [])]
    expect(not lots, f"等待拍卖中 auction_waiting: phase={st.phase} 拍品={lots}（期望 0 件）")
    # 7. 手牌叠放：按图标布局定张数；没有图标的牌（翩跹舞步）靠左边的红色角标定位
    lay = E.hand_layout(load("hand_no_icon_card.png"))
    expect(lay and lay["n"] == 1, f"无图标的牌 hand_no_icon_card: N={lay and lay['n']}（期望 1）")
    ex = StateExtractor(PROJECT_DIR / "assets" / "image")
    for name, n in (("hand_stacked_9.png", 9), ("hand_stacked_9b.png", 9), ("hand_awakened_dupes.png", 6)):
        lay = E.hand_layout(load(name))
        expect(lay and lay["n"] == n, f"手牌张数 {name}: {lay and lay['n']}（期望 {n}）")
    # 叠放时只露出一两个字：见过完整名字之后（先不叠、后叠）要能按前缀认出来
    ex = StateExtractor(PROJECT_DIR / "assets" / "image")
    ex._extract_hand(ctx, load("card_pick_fake_talent_col.png"), GameState(phase="prepare"))
    st = GameState(phase="prepare")
    ex._extract_hand(ctx, load("hand_stacked_9.png"), st)
    got = [s["name"] for s in st.hand_slots]
    cards = len([n for n in got if n != "枫叶舞"]) + len(st.hand_unread)
    expect(cards == 9 and got.count("孙悟空") == 3 and got.count("干将莫邪") == 2 and "刘邦" in got,
           f"叠放手牌 hand_stacked_9: {got} + 待看 {len(st.hand_unread)}（期望共 9 张，孙悟空×3 干将莫邪×2 刘邦）")
    return bad


def check_merge_confirm(ctx) -> list[str]:
    """合成前核对（live_match_agent.confirm_merge_target）：点开看到的不是同名就不合成、并改账本。"""
    import cv2
    import live_match_agent as L
    from agent.decision.rules import RuleEngine
    from agent.perception.extractors import StateExtractor, _imread
    from agent.perception.state import GameState
    ex = StateExtractor(PROJECT_DIR / "assets" / "image")
    panel = cv2.cvtColor(_imread(PROJECT_DIR / "data/logs/fixtures/hero_panel_blocks_levelup.png"),
                         cv2.COLOR_BGR2RGB)          # 面板上是 孙膑
    plain = cv2.cvtColor(_imread(PROJECT_DIR / "data/logs/fixtures/hand_awakened_dupes.png"),
                         cv2.COLOR_BGR2RGB)
    frames = {"img": plain}

    class FakeExec:
        def _tap(self, x, y):
            frames["img"] = plain if frames["img"] is panel else panel   # 点单位 -> 开面板；点空地 -> 关
    real_cap = L.screencap
    L.screencap = lambda _c: frames["img"]
    bad = []
    try:
        rules = RuleEngine(PROJECT_DIR / "data" / "chess_pieces.json",
                           cards_db_path=PROJECT_DIR / "data" / "cards.json")
        rules._field = {"弈星": (1027, 572)}
        st = GameState(phase="prepare", round_no=7)
        act = {"type": "drag_card", "piece": "弈星", "merge": True, "src": (631, 662), "dst": (1027, 572)}
        L._MERGE_SEEN.update({"round": -1, "cells": {}})
        ok = L.confirm_merge_target(ctx, None, FakeExec(), ex, rules, st, act)
        good = (not ok and "弈星" not in rules._field and rules._field.get("孙膑") == (1027, 572))
        print(f"[{'OK' if good else 'FAIL'}] 合成前核对：那格其实是孙膑 -> 放行={ok} 账本={rules._field}"
              "（期望 不合成、账本改成 孙膑@1027,572）")
        if not good:
            bad.append(f"合成前核对没拦住/没改账本: ok={ok} field={rules._field}")
        # 同一格本回合已经看过，第二次不再点（不花时间）
        taps = []
        class CountExec(FakeExec):
            def _tap(self, x, y):
                taps.append((x, y)); super()._tap(x, y)
        rules._field["孙膑"] = (1027, 572)
        act2 = dict(act, piece="孙膑")
        ok2 = L.confirm_merge_target(ctx, None, CountExec(), ex, rules, st, act2)
        good2 = ok2 and not taps
        print(f"[{'OK' if good2 else 'FAIL'}] 合成前核对：本回合看过的格子直接放行，没再点={not taps}")
        if not good2:
            bad.append("同一格本回合重复点开核对")
    finally:
        L.screencap = real_cap
    return bad


def main() -> int:
    frames_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        PROJECT_DIR / "data" / "logs" / "match_170447"
    )
    from agent.perception import onnx_ocr
    if onnx_ocr.enabled():
        # OCR 走 onnxruntime（默认），不需要 MaaFW 的 Tasker -> **不连模拟器**
        #（对局在跑时也能跑回归，不会和它抢设备）。WZWXQ_OCR=maa 才连模拟器测 MaaFW 的 OCR。
        ctx = None
        print(f"[OCR 后端] {onnx_ocr.OnnxOcr.get().describe()}（不连模拟器）")
    else:
        controller = connect_adb_controller()
        tasker = create_tasker(controller, PROJECT_DIR / "assets")
        ctx = FrameContext(tasker)
    extractor = StateExtractor(PROJECT_DIR / "assets" / "image")

    failures = 0
    fixtures_dir = PROJECT_DIR / "data" / "logs" / "fixtures"
    all_cases = [(frames_dir, n, v) for n, v in CASES.items()]
    all_cases += [(fixtures_dir, n, v) for n, v in EXTRA_CASES.items()]
    for base, name, (expect_phase, fields) in all_cases:
        path = base / name
        image = cv2.imread(str(path))
        if image is None:
            print(f"[SKIP] {name}: 读不到帧")
            continue
        state = extractor.extract(ctx, image)
        ok = state.phase == expect_phase
        failures += 0 if ok else 1
        print(f"[{'OK' if ok else 'FAIL'}] {name}: phase={state.phase} (期望 {expect_phase})")
        print(f"       round={state.round_no} cd={state.countdown} "
              f"gold={state.gold} energy={state.energy}/{state.energy_max} "
              f"ban={state.banned_faction!r} lvl_cost={state.level_cost} "
              f"free_refresh={state.free_refresh}")
        if "shop" in fields:
            for s in state.shop:
                print(f"       shop[{s.index}] {s.piece} cost={s.cost} faction={s.faction}")
        if "hand" in fields:
            print(f"       hand={state.hand}")
        if "players" in fields:
            print(f"       players={[p.hp for p in state.players]}")
        if "auction" in fields and state.auction:
            print(f"       auction_cd={state.auction.countdown}")
            for lot in state.auction.lots:
                print(f"       lot[{lot.index}] {lot.piece} top_bid={lot.top_bid} "
                      f"top_bidder={lot.top_bidder!r} bidders={lot.bidders} sold={lot.sold}")
        if "talents" in fields:
            for t in state.talents:
                print(f"       talent[{t.index}] {t.name}: {t.desc}")
        if "card_options" in fields:
            print(f"       card_options={[(o['name'], o['x'], o['y']) for o in state.card_options]}")
            if len(state.card_options) < 2:
                failures += 1
                print("       FAIL: N选1 选项没认全（不选就会一直卡死）")
    failures += len(check_phase_cache(ctx))
    failures += len(check_levels())
    failures += len(check_hp_filter())
    failures += len(check_player_hp(ctx))
    failures += len(check_equip_popup(ctx))
    failures += len(check_badges())
    failures += len(check_vision_units())
    failures += len(check_awaken_progress())
    failures += len(check_auction_bidders())
    failures += len(check_auction_all_lots())
    failures += len(check_battle_away())
    failures += len(check_solo_join(ctx))
    failures += len(check_level_cost_from_xp(ctx))
    failures += len(check_hand_awakened(ctx))
    failures += len(check_inspect(ctx))
    failures += len(check_board_inspect(ctx))
    failures += len(check_name_resolve())
    failures += len(check_hand_and_pop(ctx))
    failures += len(check_unit_anchors())
    failures += len(check_hero_panel(ctx))
    failures += len(check_match_0923(ctx))
    failures += len(check_merge_confirm(ctx))
    print(f"\n{'全部通过' if failures == 0 else f'{failures} 个用例失败'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

