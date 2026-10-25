"""静态检查：线上会跑到的源文件里不能有"未定义的名字"（pyflakes）。

2026-10-20 第 842 局：补丁在 laya_policy 里用了 _meta，但只有函数里的局部导入、模块顶上没导入 →
每道商店题 NameError、退回规则层只会刷新，明世隐 / 盾山 / 程咬金 / 铠 进店都没买。冒烟测试和决策重放都没跑到这条路径，
所以改完代码除了 test_rules_smoke / test_decisions_replay，再跑这一个。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
FILES = [
    "agent/decision/laya_policy.py", "agent/decision/rules.py", "agent/decision/cgnet_policy.py",
    "agent/decision/interim_rules.py", "agent/decision/talent_conditions.py", "agent/decision/hero_traits.py",
    "agent/decision/routes.py", "agent/decision/meta.py", "agent/actions/executor.py",
    "agent/perception/extractors.py", "agent/perception/state.py", "tools/live_match_agent.py",
]
# 已核实的误报：live_match_agent 主循环第一圈 state 还没赋值，那一行包在 try 里
ALLOW = {("tools/live_match_agent.py", "undefined name 'state'")}


def main() -> int:
    out = subprocess.run([sys.executable, "-m", "pyflakes", *FILES], cwd=PROJECT_DIR,
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    bad = []
    import py_compile
    for f in FILES:                      # 语法错误也算（2026-10-20：interim_rules 里中文说明夹了英文双引号，pyflakes 报了但这里只看 undefined name，漏掉）
        try:
            py_compile.compile(str(PROJECT_DIR / f), doraise=True)
        except py_compile.PyCompileError as e:
            bad.append(f"{f}: 语法错误 {str(e).splitlines()[-1][:120]}")
    for line in out.splitlines():
        if "undefined name" not in line:
            continue
        f = line.split(":")[0].replace("\\", "/")
        if any(f == a and msg in line for a, msg in ALLOW):
            continue
        bad.append(line)
    for b in bad:
        print("FAIL", b)
    print("静态检查通过（无未定义名字）" if not bad else f"静态检查失败：{len(bad)} 处未定义名字")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
