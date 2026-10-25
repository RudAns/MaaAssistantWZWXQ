# 使用方法

> 本项目仅供学习研究，不得用于商业用途（见根目录 `LICENSE`）。
> 游戏用户协议禁止第三方自动化工具，请使用小号，自行承担账号风险。

## 1. 环境要求

| 项目 | 要求 |
|---|---|
| 系统 | Windows 10 / 11 |
| Python | 3.10 或以上（开发时用 3.11） |
| 显卡 | 建议 NVIDIA（开发时用 RTX 3060 12G）；没有 N 卡也能跑，但 OCR 与 Laya 会慢很多 |
| 模拟器 | MuMu 12，分辨率 1920×1080（截图按 1280×720 基准自动换算） |
| 游戏 | 《王者万象棋》，已登录，棋手使用阿离 |

## 2. 安装

```bash
git clone https://github.com/RudAns/MaaAssistantWZWXQ.git
cd MaaAssistantWZWXQ
```

双击根目录 `setup_env.bat`：自动建 `.venv`、装依赖，可选安装 CUDA 版 torch。

手动安装：

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
# N 卡加速（可选）：OCR 走 TensorRT，整帧约 0.1 秒
.venv/Scripts/python.exe -m pip uninstall -y onnxruntime
.venv/Scripts/python.exe -m pip install -r requirements-nvidia.txt
# CUDA 版 torch（可选，Laya 推理从秒级降到约 80 ms）
.venv/Scripts/python.exe -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch
```

第一次使用 TensorRT 时要为本机显卡编译引擎，约 8 分钟，之后读缓存。

## 3. 下载训练好的模型

模型不在代码库里，放在 GitHub Releases（页面右侧 Releases → 最新版本）。下载三个压缩包，**在仓库根目录解压**（压缩包内已带好相对路径）：

| 压缩包 | 内容 | 解压后位置 |
|---|---|---|
| `models-perception.zip` | OCR（找字轻量版 + 认字服务器版）、场上单位检测、等级、觉醒进度、认人、装备件数、手牌条、手牌立绘、手牌觉醒、界面分类 | `assets/model/` |
| `models-laya-R52.zip` | 决策学生 Laya 第 52 版 | `data/laya/runs/R52_s1_top8/checkpoint/` |
| `models-cgnet.zip` | 局面网络（换主C 头、经济头）与冻结的卡牌图谱 | `data/cgnet/deploy/` |

Laya 的底座编码器 `jhu-clsp/mmBERT-base` 第一次运行时会从 Hugging Face 自动下载（需联网）。

## 4. 连接模拟器

1. 启动 MuMu，打开游戏并登录。
2. 在 `interface.json` 里把 adb 路径和端口改成本机的（MuMu 12 单开默认 16384，多开实例为 16416 / 16448 …），细节见 [adb-config-new-machine.md](adb-config-new-machine.md)。

## 5. 检查环境

```bash
# 规则与决策离线测试（不需要模拟器）
.venv/Scripts/python.exe -X utf8 tools/test_static_names.py
.venv/Scripts/python.exe -X utf8 tools/test_rules_smoke.py
```

## 6. 打一局

停在游戏大厅，运行：

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -X utf8 -u tools/live_match_agent.py --execute --enter-match \
  --laya-ckpt data/laya/runs/R52_s1_top8/checkpoint \
  --laya-kinds card_pick,talent,lord_boost,shop,carry,swap,auction,use_card,route,sell_hand \
  --eco-takeover
```

- 不加 `--execute` 是"只看不点"的干跑模式，最安全，适合先确认识别正常。
- 程序会自动进入对局、一直打到结算页后停止操作，**但进程不会自己退出**，开下一局前请手动结束它。
- 加 `--perf` 可以逐帧打印耗时。

## 7. 看决策过程

```bash
.venv/Scripts/python.exe -X utf8 tools/laya_dashboard/server.py --port 8765
```

浏览器打开 <http://localhost:8765>，可以实时看到 Laya 的每一次判断、各选项概率与当时的局面。

卡牌关系图谱：用浏览器打开 `docs/card_graph.html`（也可以从 Releases 下载 `card-graph.html`）。研究报告：用浏览器打开 `docs/index.html`（或从 Releases 下载 `wanxiangqi-research.html`）。

## 8. 复盘一局

```bash
.venv/Scripts/python.exe -X utf8 tools/check_components.py <局号>   # 各模型本局是否真的起作用
.venv/Scripts/python.exe -X utf8 tools/review_focus.py <局号>       # 天赋、主C、核心阵容等重点
.venv/Scripts/python.exe -X utf8 tools/blunder_check.py <局号>      # 疑似失误清单
```

局号与运行记录的对应关系写在 `data/laya/iterations/batch2_matches.jsonl`（本机生成，不在仓库里）。
