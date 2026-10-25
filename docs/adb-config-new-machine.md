# 新机器适配：修改 interface.json 的 adb_path / address

> 本文是写给 AI 助手（或新上手的人）的操作指引：在一台新机器上克隆本项目后，
> 把 `interface.json` 里写死的 MuMu 模拟器 ADB 配置改成本机实际值。
> 读完本文即可独立完成，不需要额外背景。

## 背景（为什么只需要改这两个字段）

`interface.json` 是 MaaFramework 的项目入口配置，**只有 MaaPiCli / 通用 UI 加载项目时读它**。
Agent 本体（`agent/adb_client.py`）靠 `Toolkit.find_adb_devices()` **自动发现**模拟器，
不吃这两个字段。所以这次修改只影响"用 MaaPiCli 跑任务"这条链路。

要改的位置（`interface.json` 第 12-13 行，`controller[0].adb` 下）：

```json
"adb_path": "D:/Program Files/Netease/MuMu/nx_main/adb.exe",
"address": "127.0.0.1:16384",
```

- `adb_path`：MuMu 自带的 `adb.exe` 的**绝对路径**（JSON 里用正斜杠 `/`）
- `address`：模拟器实例的 ADB 连接地址

第二个 controller（Win32）已弃用，**不要动**。其余字段一律不要动。

## 第 1 步：找到本机的 adb.exe

MuMu 12 的 adb.exe 固定在安装目录的 `nx_main\adb.exe`。常见路径：

```
C:\Program Files\Netease\MuMu\nx_main\adb.exe
D:\Program Files\Netease\MuMu\nx_main\adb.exe
```

不确定时用命令找（Git Bash）：

```bash
for d in C D E F G H I; do ls "$d:/Program Files/Netease/MuMu/nx_main/adb.exe" 2>/dev/null; done
```

或在 PowerShell 里：

```powershell
Get-ChildItem -Path C:\,D:\,E:\ -Filter adb.exe -Recurse -Depth 4 -ErrorAction SilentlyContinue |
    Where-Object FullName -like '*MuMu*'
```

如果用户装的是 MuMu Player 国际版，目录名可能是 `MuMu Player`，同样在 `shell` 或 `nx_main` 子目录下找。

## 第 2 步：确认模拟器的 ADB 端口

MuMu 12 端口规律：**第一个实例 16384**，多开实例依次 +32（16416、16448…）。

最可靠的做法——**先启动模拟器**，然后用第 1 步找到的 adb.exe 问它：

```bash
"D:/Program Files/Netease/MuMu/nx_main/adb.exe" devices
```

输出里 `127.0.0.1:xxxxx  device` 那一行的端口号就是 `address` 要填的值。

也可以让项目自己的环境帮你确认（`.venv` 已由 `setup_env.bat` 建好）：

```bash
.venv/Scripts/python.exe -X utf8 -c "from maa.toolkit import Toolkit; [print(d.adb_path, d.address) for d in Toolkit.find_adb_devices()]"
```

这个调用直接打印 MaaFramework 自动发现到的 `adb_path` 和 `address`，照抄即可。
**注意它偶尔会长时间阻塞（在扫描模拟器），等 30 秒没输出就 Ctrl+C 改用上面 adb devices 的方法。**

## 第 3 步：改 interface.json

只改 `controller[0].adb` 下的两个值，注意：

- 路径用**正斜杠**（`D:/Program Files/...`），不要用反斜杠（JSON 转义容易出错）
- 保持 JSON 结构合法，改完用 `python -m json.tool interface.json` 验证一下

## 第 4 步：验证连接

模拟器保持启动状态，跑项目自带的 ADB 连接测试：

```bash
.venv/Scripts/python.exe -X utf8 tools/adb_test.py
```

能截到图（生成 `data/logs/adb_test.png`）就说明 ADB 链路通了。

## 常见错误

| 现象 | 原因 |
|---|---|
| `adb devices` 无输出 / 空列表 | 模拟器没启动，或 MuMu 的 ADB 调试没开（MuMu 12 默认开） |
| 端口连不上 | 多开实例端口不是 16384，用 `adb devices` 实测，别猜 |
| MaaPiCli 报找不到 adb | `adb_path` 用了反斜杠或盘符不对，回第 1 步 |
