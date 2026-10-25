@echo off
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
%~dp0..\.venv\Scripts\python.exe tools\bg_input_test.py
