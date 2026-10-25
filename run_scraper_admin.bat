@echo off
:: 以管理员权限运行卡牌数据库爬虫（双击后请批准 UAC 提示）
:: 游戏必须停在 百科 页面再运行
cd /d "%~dp0"
powershell -NoProfile -Command "Start-Process -FilePath 'cmd.exe' -ArgumentList '/k set PYTHONIOENCODING=utf-8 ^&^& %~dp0.venv\Scripts\python.exe tools\update_card_db.py' -WorkingDirectory '%~dp0' -Verb RunAs"
