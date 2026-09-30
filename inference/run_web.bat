@echo off
chcp 65001 >nul
REM ============================================================
REM  NileTTS-4h-Inference — مشغل واجهة الويب
REM  يفتح صفحة في المتصفح تكتب فيها النص وتضبط كل المعاملات.
REM  خيارات:  run_web.bat --port 9000 --threads 4 --no-browser
REM ============================================================
cd /d "%~dp0"
python webapp.py %*
echo.
pause
