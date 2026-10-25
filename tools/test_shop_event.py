"""商店题"局面变了才问"的离线检查（2026-10-09）：不加载 Laya，只验触发条件与签名缓存。
.venv/Scripts/python.exe -X utf8 tools/test_shop_event.py"""
import sys
sys.path.insert(0, ".")
from agent.decision.rules import RuleEngine
from agent.decision.laya_policy import LayaPolicy
from agent.perception.state import GameState, ShopSlot

r = RuleEngine("data/chess_pieces.json", cards_db_path="data/cards.json"); r.shop_by_laya = True
assert r._buy_cap(GameState()) >= 30, "商店交给 Laya 时不按张数卡"
P = object.__new__(LayaPolicy); P.rules = r; P.kinds = {"shop"}
calls = []
P._shop_decide = lambda st, a: (calls.append(1), {"type": "buy_piece", "slot": 0})[1]
s = GameState(); s.phase = "prepare"; s.shop_open = True; s.round_no = 6; s.gold = 12
s.shop = [ShopSlot(index=0, piece="张良", cost=3)]
r._field = {"百里守约": (276, 501)}
assert P._shop_event_ok(s, None) and P._shop_event_ok(s, {"type": "toggle_shop", "intent": "close"})
assert not P._shop_event_ok(s, {"type": "drag_card"})
P._shop(s, None); P._shop(s, None)
assert len(calls) == 1, "局面没变不重问"
s.gold = 9; P._shop(s, None)
assert len(calls) == 2, "钱变了要重问"
r._stop_shop_round = 6
assert not P._shop_event_ok(s, None), "Laya 说停手后本回合不再问"
r._stop_shop_round = -1; r._field = {}
s.hand_slots = [{"name": "百里守约"}]
print("商店题触发检查全部通过")

# 马上要被换掉的老主C 不买（2026-10-09 第 473 局：拍卖拿到守约后买了干将莫邪）
r2 = RuleEngine("data/chess_pieces.json", cards_db_path="data/cards.json")
r2.route_locked = lambda: True
r2.route_carry = lambda: "百里守约"
r2.route_stage_members = lambda rnd: {"百里守约", "李元芳"}
r2._round_seen = 4; r2._carry_piece = "干将莫邪"; r2._field = {"干将莫邪": (300, 569)}
assert not r2.outgoing_unit("干将莫邪"), "路线主C 还没到手：老主C 照常"
r2._hand_book = {"百里守约": 1}
assert r2.outgoing_unit("干将莫邪") and r2.recently_sold("干将莫邪"), "守约在手：别再买干将"
assert not r2.outgoing_unit("李元芳"), "路线这阶段要的人不算"
print("老主C 不买检查通过")
