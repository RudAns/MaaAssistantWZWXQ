# 装备四选一避开已穿的同名成品（2026-10-03 规则：一个英雄不能携带两个相同的装备，4 选 1 时要避开）。不连模拟器。
import sys, json, types
from pathlib import Path
sys.path.insert(0,'.'); sys.path.insert(0,'tools')
import live_match_agent as L
R=json.loads(Path('data/equip_recipes.json').read_text(encoding='utf-8'))['recipes']
base='风暴巨剑'; rec=R[base]
ext=types.SimpleNamespace(equip_recipes=R, image_dir=Path('assets/image'))
L.find_equip_slots=lambda img,c:[(400+i*60,400) for i in range(4)]
L.find_equip_thumb=lambda img,slots:None
p0=L.detect_equip_popup(None,None,ext,(500,500),'carry:亚连',None,base)
first=L._LAST_FORGED
p1=L.detect_equip_popup(None,None,ext,(500,500),'carry:亚连',None,base,worn={first})
print(rec,'| 无已穿:',first,'| 已穿它:',L._LAST_FORGED)
assert L._LAST_FORGED!=first and L._LAST_FORGED in rec
L.find_equip_thumb=lambda img,slots:1
L.detect_equip_popup(None,None,ext,(500,500),'tank',None,base); t=L._LAST_FORGED
L.detect_equip_popup(None,None,ext,(500,500),'tank',None,base,worn={t}); print('👍:',t,'-> 已穿后',L._LAST_FORGED); assert L._LAST_FORGED!=t
print('OK')
