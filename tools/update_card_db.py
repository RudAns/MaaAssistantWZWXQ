"""卡牌数据库爬虫：遍历游戏内百科页，自动重建卡牌数据库。

原理：
  百科 → 各标签页（英雄牌/效果牌/装备牌/天赋牌/棋手牌）→ 网格 OCR 定位名称标签
  → 点击标签上方头像 → OCR 右侧详情面板（名称/派系/标签/技能）→ 裁剪头像存档。

游戏版本更新后重新运行本工具即可重建数据库，全程不依赖 AI。

用法:
  python tools/update_card_db.py                 # 全量爬取（自动从主界面导航进百科）
  python tools/update_card_db.py --tabs 英雄牌    # 只爬指定标签页
  python tools/update_card_db.py --dry-run       # 只 OCR 网格并保存标注图，不点击

坐标基准：1186x720 窗口（游戏 PC 版实际截图分辨率）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.win32_client import connect_controller, create_tasker, ocr_once, screencap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
DB_PATH = ROOT / "data" / "cards.json"
PORTRAIT_DIR = ASSETS / "image" / "pieces"

# ---- 百科页布局坐标（按后端区分）----
def _win32_layout() -> dict:
    """PC 版 1186x720 实测校准。"""
    return dict(
        TABS={"英雄牌": (385, 83), "效果牌": (522, 83), "装备牌": (658, 83),
              "天赋牌": (795, 83), "棋手牌": (933, 83)},
        ROI_GRID=(30, 150, 905, 510),
        ROI_DETAIL_NAME=(985, 240, 180, 60),
        ROI_DETAIL_CONTENT=(940, 300, 250, 360),
        ROI_COLLECT=(790, 105, 220, 35),
        DETAIL_TABS={"说明": (966, 684), "技能": (1014, 684),
                     "属性": (1061, 684), "觉醒": (1115, 684)},
        MENU_BUTTON=(24, 22),
        SCROLL_STEP=((500, 470), (500, 345)),
        SCROLL_TO_TOP=((500, 250), (500, 550)),
        SCROLL_BACK=((500, 345), (500, 415)),
        PORTRAIT_W=90, PORTRAIT_H=86, CLICK_OFFSET_Y=60,
        SCREEN_W=1186,
    )


def _adb_layout() -> dict:
    """模拟器 1280x720 实测校准（dry-run 标注图核对）。"""
    return dict(
        TABS={"英雄牌": (417, 106), "效果牌": (561, 106), "装备牌": (708, 106),
              "天赋牌": (853, 106), "棋手牌": (1001, 106)},
        ROI_GRID=(30, 145, 955, 520),
        ROI_DETAIL_NAME=(1040, 320, 210, 50),
        ROI_DETAIL_CONTENT=(1000, 385, 240, 290),
        ROI_COLLECT=(800, 128, 180, 35),
        DETAIL_TABS={"说明": (1014, 688), "技能": (1075, 688),
                     "属性": (1135, 688), "觉醒": (1197, 688)},
        MENU_BUTTON=(28, 25),
        SCROLL_STEP=((540, 470), (540, 345)),
        SCROLL_TO_TOP=((540, 250), (540, 550)),
        SCROLL_BACK=((540, 345), (540, 415)),
        PORTRAIT_W=90, PORTRAIT_H=86, CLICK_OFFSET_Y=60,
        SCREEN_W=1280,
    )


LAYOUTS = {"win32": _win32_layout(), "adb": _adb_layout()}

# 当前生效布局（main() 按 --backend 注入）
TABS: dict = {}
ROI_GRID = ROI_DETAIL_NAME = ROI_DETAIL_CONTENT = ROI_COLLECT = (0, 0, 0, 0)
DETAIL_TABS: dict = {}
MENU_BUTTON = (0, 0)
SCROLL_STEP = SCROLL_TO_TOP = SCROLL_BACK = ((0, 0), (0, 0))
PORTRAIT_W, PORTRAIT_H, CLICK_OFFSET_Y = 90, 86, 60
SCREEN_W = 1186

MAX_PASSES = 60

_CJK = re.compile(r"^[一-鿿]{1,5}$")

# 详情面板页签词，OCR 误识别时不能当作卡名
UI_WORDS = {"说明", "技能", "属性", "觉醒", "历史", "出装", "已收集"}

# 详情面板底部页签坐标（1186x720 实测）：说明/技能/属性/觉醒
DETAIL_TABS = {
    "说明": (966, 684),
    "技能": (1014, 684),
    "属性": (1061, 684),
    "觉醒": (1115, 684),
}
ROI_DETAIL_CONTENT = (940, 300, 250, 360)   # 页签内容区（说明/技能/属性/觉醒共用）
ROI_COLLECT = (790, 105, 220, 35)           # 已收集 x/y

# ---- 主界面导航 ----
MENU_BUTTON = (24, 22)                      # 主界面左上角菜单按钮（≡）
NAV_MAX_RETRY = 3

# 合法标签词表（OCR 去噪用；"三"前缀是列表图标误读；"出装"是按钮不是标签，不入表）
TAG_VOCAB = {"前排", "后排", "坦克", "法师", "射手", "刺客", "辅助", "战士"}

_FACTION_LINE = re.compile(r"^[-·—\s]*([一-鿿]{2,6})[-·—\s]*$")


def parse_tags(text: str) -> list[str]:
    """从标签行 OCR 文本提取合法标签（去图标噪声'三'前缀）。"""
    tags = []
    for line in text.split("\n"):
        line = line.strip().lstrip("三≡=").strip()
        if line in TAG_VOCAB and line not in tags:
            tags.append(line)
    return tags


def parse_faction(lines: list[str]) -> str:
    """从说明页内容行中找派系行（形如 -大河流域- / ·河洛·）。"""
    for line in lines:
        stripped = line.strip()
        core = stripped.strip("-·— \t")
        if stripped != core and _CJK.match(core) and len(core) >= 2:
            return core
    return ""


def scrape_detail_tabs(controller, tasker, tabs: dict | None = None) -> dict[str, str]:
    """依次点击详情面板的页签，OCR 各自内容区。

    tabs 默认英雄牌的四页签；pos 为 None 表示不点击、直接 OCR 当前面板
    （效果牌/装备牌等没有底部页签，卡面即全部信息）。
    返回 {页签名: 原始文本}；内容与其他页签重复的（该卡无此页签）不记录。
    """
    if tabs is None:
        tabs = DETAIL_TABS
    out: dict[str, str] = {}
    for tab_name, pos in tabs.items():
        if pos is not None:
            controller.post_click(*pos).wait()
            time.sleep(0.6)
        img = screencap(controller)
        text = ocr_once(tasker, img, ROI_DETAIL_CONTENT, multi=True)
        if text and text not in out.values():
            out[tab_name] = text
    return out


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    """cv2.imwrite 不支持中文路径，用 imencode 绕过。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cv2.imencode(".png", image)[1].tobytes())


def grid_ocr(tasker, image, roi=None) -> list[dict]:
    """OCR 指定区域，返回 [{text, box, score}]（box 为绝对坐标）。"""
    if roi is None:
        roi = ROI_GRID
    node = {"recognition": "OCR", "roi": list(roi), "action": "DoNothing"}
    task_detail = tasker.post_task("__grid_ocr__", {"__grid_ocr__": node}).wait().get()
    out = []
    if task_detail is None:
        return out
    for nd in task_detail.nodes:
        reco = nd.recognition
        if reco and reco.hit:
            for r in (reco.filtered_results or reco.all_results):
                out.append({"text": r.text.strip(), "box": list(r.box), "score": r.score})
    return out


def navigate_to_baike(controller, tasker) -> bool:
    """从主界面导航进百科：点左上角菜单 → 全屏 OCR 找"百科"入口 → 点击进入并校验页头。

    任意界面下可调用；若已在百科页则直接返回 True。
    """
    head_roi = (20, 15, 120, 45)
    if "百科" in ocr_once(tasker, screencap(controller), head_roi):
        print("  导航: 已在百科页", flush=True)
        return True
    for attempt in range(1, NAV_MAX_RETRY + 1):
        controller.post_click(*MENU_BUTTON).wait()
        time.sleep(1.2)
        blocks = grid_ocr(tasker, screencap(controller), (0, 0, SCREEN_W, 720))
        target = next((b for b in blocks if b["text"] == "百科"), None)
        if target is None:
            print(f"  导航: 第{attempt}次未找到'百科'菜单项，"
                  f"OCR 到: {[b['text'] for b in blocks][:15]}", flush=True)
            controller.post_click(*MENU_BUTTON).wait()   # 收起菜单复位
            time.sleep(0.8)
            continue
        bx, by, bw, bh = target["box"]
        controller.post_click(bx + bw // 2, by + bh // 2).wait()
        time.sleep(1.5)
        if "百科" in ocr_once(tasker, screencap(controller), head_roi):
            print("  导航: 已进入百科页", flush=True)
            return True
        print(f"  导航: 第{attempt}次点击后未到达百科页，重试", flush=True)
    return False


_LABEL_RE = re.compile(r"^([一-鿿0-9]{1,6})…?$")   # 效果牌等名字带省略号截断（如"高阶招募…"）


def filter_name_labels(items: list[dict]) -> list[dict]:
    """从网格 OCR 结果中筛出卡牌名称标签：中文/数字 1-6 字（允许省略号结尾）、排除派系标题（更高的框）。"""
    labels = []
    for it in items:
        text, box = it["text"], it["box"]
        if it["score"] < 0.8:               # 阈值 0.8：卡名是标准字体，低分也多是真标签；过滤靠字形/高度兜底
            continue
        m = _LABEL_RE.match(text)
        if not m or not re.search(r"[一-鿿]", text):
            continue
        core = m.group(1)
        if core in UI_WORDS:
            continue
        if box[3] > 19:                     # 派系标题（如"大河流域"）字号更大、框更高
            continue
        it = dict(it)
        it["text"] = core                   # 省略号去掉，后续用前缀匹配详情大名
        labels.append(it)
    return labels


def drag(controller, src: tuple[int, int], dst: tuple[int, int], steps: int = 20,
         step_delay: float = 0.02) -> None:
    """拟人拖拽：touch_down → 多个中间点 touch_move → touch_up。"""
    controller.post_touch_down(src[0], src[1]).wait()
    time.sleep(0.1)
    for i in range(1, steps + 1):
        x = int(src[0] + (dst[0] - src[0]) * i / steps)
        y = int(src[1] + (dst[1] - src[1]) * i / steps)
        controller.post_touch_move(x, y).wait()
        time.sleep(step_delay)
    controller.post_touch_up().wait()
    time.sleep(0.5)


def tab_db_path(tab_name: str) -> Path:
    return ROOT / "data" / "cards" / f"{tab_name}.json"


def scrape_tab(controller, tasker, tab_name: str, dry_run: bool) -> dict:
    """爬取一个标签页，返回 {卡牌名: 记录}。

    增量写入 data/cards/<标签页>.json：每收录一张立即落盘，中断不丢数据；
    重跑时自动跳过已收录的卡。
    """
    print(f"\n=== 标签页: {tab_name} ===", flush=True)
    controller.post_click(*TABS[tab_name]).wait()
    time.sleep(1.5)
    if not dry_run:
        for _ in range(4):                  # 回到顶部
            drag(controller, *SCROLL_TO_TOP)
            time.sleep(0.3)

    # 读取"已收集 x/y"作为数量验收基准
    expect_text = ocr_once(tasker, screencap(controller), ROI_COLLECT)
    m = re.search(r"(\d+)\s*/\s*(\d+)", expect_text)
    expected_total = int(m.group(2)) if m else 0
    print(f"  数量验收基准: {expect_text!r} -> {expected_total}", flush=True)

    cards: dict[str, dict] = {}
    db_path = tab_db_path(tab_name)
    if not dry_run and db_path.exists():
        cards = json.loads(db_path.read_text(encoding="utf-8")).get("cards", {})
        if cards:
            print(f"  已有 {len(cards)} 张记录，将跳过并增量补充", flush=True)
    tab_dir = PORTRAIT_DIR / tab_name

    def save() -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.write_text(json.dumps(
            {"version": time.strftime("%Y-%m-%d"), "expected": expected_total,
             "cards": cards},
            ensure_ascii=False, indent=2), encoding="utf-8")

    last_sig: tuple | None = None
    faction_hint: dict[str, str] = {}
    backtracks = 0
    for pass_no in range(MAX_PASSES):
        image = screencap(controller)
        items = grid_ocr(tasker, image)
        labels = filter_name_labels(items)
        labels.sort(key=lambda it: (it["box"][1] // 100, it["box"][0]))  # 列表顺序：先行后列

        # 派系标题（靠左对齐且文本框更高）按 y 归属给下方的标签，兜底详情页派系解析失败
        marked = [(it, it["box"][3] > 19 and it["box"][0] < 120) for it in items
                  if it["score"] >= 0.8 and _CJK.match(it["text"]) and it["text"] not in UI_WORDS]
        current_section = ""
        for it, is_header in sorted(marked, key=lambda m: m[0]["box"][1]):
            if is_header:
                current_section = it["text"]
            elif current_section:
                faction_hint.setdefault(it["text"], current_section)

        sig = tuple((it["text"], it["box"][1] // 20) for it in labels)
        print(f"[pass {pass_no}] OCR {len(items)} 个文本块，名称标签 {len(labels)} 个")

        if dry_run:
            annotated = image.copy()
            for it in labels:
                bx, by, bw, bh = it["box"]
                cx = bx + bw // 2                       # 以标签中心对准头像中心
                cv2.rectangle(annotated, (cx - PORTRAIT_W // 2, by - CLICK_OFFSET_Y - PORTRAIT_H // 2),
                              (cx + PORTRAIT_W // 2, by - CLICK_OFFSET_Y + PORTRAIT_H // 2),
                              (0, 255, 0), 2)
                cv2.circle(annotated, (cx, by - CLICK_OFFSET_Y), 4, (0, 0, 255), -1)
            out = ROOT / "data" / "logs" / f"dryrun_{tab_name}_{pass_no}.png"
            imwrite_unicode(out, annotated)
            print(f"  标签: {[it['text'] for it in labels]}")
            print(f"  dry-run 标注图: {out}")
            break

        new_in_pass = 0
        for it in labels:
            bx, by, bw, bh = it["box"]
            label = it["text"]
            if label in cards:
                continue

            click_x, click_y = bx + bw // 2, by - CLICK_OFFSET_Y

            # 点击并校验详情面板确实切换到了这张卡，不匹配则重试
            name = ""
            detail_img = None
            for attempt in range(3):
                controller.post_click(click_x, click_y).wait()
                time.sleep(1.0 + attempt * 0.5)
                detail_img = screencap(controller)
                candidate = ocr_once(tasker, detail_img, ROI_DETAIL_NAME)
                if candidate and candidate not in UI_WORDS and (
                        candidate == label or label.startswith(candidate)
                        or candidate.startswith(label)):
                    name = candidate
                    break
                if candidate and candidate in cards:
                    # 点到的卡已收录，没必要重试
                    break
            if not name:
                print(f"  ! {label} 点击后详情未匹配（3 次重试失败），留待补爬", flush=True)
                continue
            if name in cards:
                continue

            # 英雄牌有四页签（说明/技能/属性/觉醒）；效果牌等卡面即全部信息，直接 OCR 当前面板
            tabs = scrape_detail_tabs(controller, tasker,
                                      None if tab_name == "英雄牌" else {"说明": None})
            intro_lines = tabs.get("说明", "").split("\n")

            # 派系行与标签行之外的正文视为技能描述
            faction = parse_faction(intro_lines) or faction_hint.get(label, "")
            tags = parse_tags(tabs.get("说明", ""))
            skill_lines = [
                ln for ln in intro_lines
                if ln.strip() and parse_faction([ln]) == ""
                and ln.strip().lstrip("三≡=").strip() not in TAG_VOCAB
            ]

            record = {
                "name": name,
                "category": tab_name,
                "label_text": label,
                "faction": faction,
                "tags": tags,
                "skill": "\n".join(skill_lines),
                "tabs": tabs,
                "portrait": f"image/pieces/{tab_name}/{name}.png",
            }

            # 从头像区域裁剪存档（以标签中心对准头像中心）
            cx = bx + bw // 2
            px, py = cx - PORTRAIT_W // 2, click_y - PORTRAIT_H // 2
            portrait = image[max(0, py):py + PORTRAIT_H, max(0, px):px + PORTRAIT_W]
            imwrite_unicode(tab_dir / f"{name}.png", portrait)

            cards[name] = record
            new_in_pass += 1
            save()                          # 每张卡立即落盘
            print(f"  + {name}  [{record['faction']}]  {record['tags']}", flush=True)

        # 到底判定：滚动后视图与上一轮完全一致
        if last_sig is not None and sig == last_sig:
            print("  视图不再变化，判定已到列表底部", flush=True)
            break
        # 跳行检测：与上一轮零重叠说明一次滚过了头，回滚半行重看
        if last_sig is not None and sig and backtracks < 5:
            prev_texts = {t for t, _ in last_sig}
            if not (prev_texts & {t for t, _ in sig}):
                backtracks += 1
                print("  相邻两屏零重叠，可能跳行，回滚半行重看", flush=True)
                drag(controller, *SCROLL_BACK)
                time.sleep(0.8)
                last_sig = None           # 避免把回滚后的视图误判为到底
                continue
        last_sig = sig
        drag(controller, *SCROLL_STEP)      # 向下滚动约一行
        time.sleep(0.8)

    print(f"=== {tab_name} 完成，共 {len(cards)} 张 ===", flush=True)
    if expected_total and len(cards) < expected_total:
        print(f"  警告: 收录 {len(cards)}/{expected_total}，有遗漏，"
              f"建议重跑本标签页（已收录的会自动跳过）", flush=True)
    return cards


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["adb", "win32"], default="adb",
                        help="adb=模拟器（完全后台，推荐）；win32=PC 客户端（需管理员，占用鼠标）")
    parser.add_argument("--tabs", default=None, help="逗号分隔的标签页，默认全部")
    parser.add_argument("--dry-run", action="store_true", help="只 OCR 网格并保存标注图")
    args = parser.parse_args()

    globals().update(LAYOUTS[args.backend])     # 注入当前后端的布局坐标
    tabs_arg = args.tabs or ",".join(TABS)

    if args.backend == "adb":
        from agent.adb_client import connect_adb_controller
        controller = connect_adb_controller()
    else:
        controller = connect_controller()
    tasker = create_tasker(controller, ASSETS)

    # 从当前界面自动导航进百科（主界面 → 左上角菜单 → 百科）
    if not navigate_to_baike(controller, tasker):
        print("错误：无法导航到百科页，请确认游戏已启动并处于主界面")
        return 1

    for tab in tabs_arg.split(","):
        tab = tab.strip()
        if tab not in TABS:
            print(f"未知标签页: {tab}，可选: {list(TABS)}")
            continue
        # 自动补漏：若收录数未达到"已收集 x/y"的 y，重爬该页（已收录的自动跳过），最多 3 轮
        for round_no in range(3):
            cards = scrape_tab(controller, tasker, tab, args.dry_run)
            if args.dry_run:
                break
            print(f"  已保存: {tab_db_path(tab)}（{len(cards)} 张）", flush=True)
            p = tab_db_path(tab)
            expected = 0
            if p.exists():
                expected = json.loads(p.read_text(encoding="utf-8")).get("expected", 0)
            if not expected or len(cards) >= expected:
                break
            print(f"  收录 {len(cards)}/{expected}，第 {round_no + 2} 轮补爬...", flush=True)

    if not args.dry_run:
        total = 0
        for tab in TABS:
            p = tab_db_path(tab)
            if p.exists():
                total += len(json.loads(p.read_text(encoding="utf-8")).get("cards", {}))
        print(f"\n全部完成，数据库目录: {ROOT / 'data' / 'cards'}（共 {total} 张卡）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
