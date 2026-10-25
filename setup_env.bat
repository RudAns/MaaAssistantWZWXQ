@echo off
rem ============================================================
rem  MaaAssistantWZWXQ 开发环境一键还原
rem  从 GitHub 克隆后双击运行（普通权限即可，不需要管理员）
rem  还原内容：.venv 虚拟环境 + pip 依赖 + (可选) CUDA torch
rem            + (可选) Laya 模型权重 ~3.5G 到 .hf_cache
rem  不还原：data/logs 对局录像、debug 日志（本机产物，无需还原）
rem ============================================================
setlocal
cd /d "%~dp0"

echo === [1/5] 检查 Python 3.10+ ===
where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 python，请先安装 Python 3.10+ 并加入 PATH
    exit /b 1
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
    echo [错误] Python 版本过低，需要 3.10+
    exit /b 1
)

echo === [2/5] 创建虚拟环境 .venv ===
if exist .venv\Scripts\python.exe (
    echo .venv 已存在，跳过
) else (
    python -m venv .venv || exit /b 1
)
set PY=.venv\Scripts\python.exe

echo === [3/5] 安装依赖 requirements.txt ===
%PY% -m pip install --upgrade pip || exit /b 1
%PY% -m pip install -r requirements.txt || exit /b 1

echo === [3b] N 卡 OCR 加速 TensorRT（可选，只有 N 卡能用）===
echo 识别一帧 377ms -^> 91ms。A 卡 / 核显 / 没独显跳过即可，会自动用 CPU 版，功能一样只是慢些
echo 第一次运行 agent 要为本机显卡建引擎，约 8 分钟，只需一次（缓存在 .trt_cache）
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo 没检测到 NVIDIA 显卡，跳过
) else (
    choice /c YN /n /m "安装 TensorRT OCR 加速? [Y/N]: "
    if not errorlevel 2 (
        %PY% -m pip uninstall -y onnxruntime
        %PY% -m pip install -r requirements-nvidia.txt || exit /b 1
    )
)

echo === [4/5] GPU 版 PyTorch（可选，有 N 卡强烈建议装）===
echo PyPI 默认 torch 在 Windows 上是 CPU 版，CUDA 版可让 Laya 推理从秒级降到几十毫秒
choice /c YN /n /m "安装 CUDA 12.8 版 torch? [Y/N]: "
if not errorlevel 2 (
    %PY% -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch || exit /b 1
)

echo === [5/5] Laya 模型权重（可选，约 3.5G，下载到 .hf_cache）===
echo 只下 multilingual + typed-decisions，不含 english（项目用不到）
choice /c YN /n /m "现在下载 Laya 权重? [Y/N]: "
if not errorlevel 2 (
    %PY% -m pip install huggingface_hub || exit /b 1
    set HF_HOME=%~dp0.hf_cache
    %PY% -c "from huggingface_hub import snapshot_download; snapshot_download('convaiinnovations/laya', ignore_patterns=['english/*']); print('laya weights ok')" || exit /b 1
    rem 把 HF_HOME 写成用户级环境变量，以后不用每次设
    powershell -NoProfile -Command "[Environment]::SetEnvironmentVariable('HF_HOME','%~dp0.hf_cache','User')"
    echo 已设置 HF_HOME（当前账户环境变量）=%~dp0.hf_cache
)

echo === 验证 ===
%PY% -c "import maa, numpy, cv2, laya; print('deps ok')" || exit /b 1
echo 检查 OCR 引擎（装了 TensorRT 的话，第一次会在这里建引擎，约 8 分钟）
%PY% -c "from agent.perception import onnx_ocr as O; print('OCR 引擎:', O.OnnxOcr.get().describe())" || exit /b 1

echo.
echo 环境还原完成。还差两样只能手动装的：
echo   1. MuMu 12 模拟器 + 《王者万象棋》（见 docs/usage.md）
echo   2. interface.json 里的 adb_path / address 按本机实际改（指引见 docs/adb-config-new-machine.md）
echo 验证环境：.venv\Scripts\python.exe -X utf8 tools\test_rules_smoke.py
endlocal
