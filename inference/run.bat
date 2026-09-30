@echo off
REM ============================================================
REM  NileTTS-4h-Inference - مشغل سريع على Windows
REM  مثال:  run.bat --text "السلام عليكم" --out out.wav
REM  يمرر كل الوسائط إلى infer.py كما هي.
REM ============================================================
cd /d "%~dp0"
python infer.py %*
if errorlevel 1 (
  echo.
  echo [خطأ] فشل التوليد — انظر الرسالة أعلاه.
  pause
)
