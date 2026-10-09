@echo off
chcp 65001 >nul
REM ============================================================
REM  بناء نماذج ONNX لتجربة المتصفح — أمر واحد على Windows
REM  المتطلبات المسبقة:
REM    1) checkpoint الإنتاج: inference\checkpoints\states_79590.pth
REM    2) python -m pip install -r requirements.txt
REM       python -m pip install -r inference\onnx\requirements-onnx.txt
REM ============================================================
cd /d "%~dp0.."
echo [1/4] تصدير FP32 + فحص التكافؤ مقابل PyTorch ...
python onnx\export_onnx.py --checkpoint states_79590.pth
if errorlevel 1 goto :err
echo.
echo [2/4] التكميم (FP16 / INT8 ديناميكي / INT8 ثابت) + القياس ...
python onnx\quantize_onnx.py
if errorlevel 1 goto :err
echo.
echo [3/4] توليد بيانات خط النص للمتصفح + مراجع التكافؤ ...
python onnx\gen_textpipe_data.py
if errorlevel 1 goto :err
echo.
echo [4/4] قياس أداء الخادم لكل نسخة ...
python onnx\bench_onnx.py
if errorlevel 1 goto :err
echo.
echo ============================================================
echo  تم البناء — شغّل التجربة:
echo      cd ..\web-exp
echo      python serve.py
echo  ثم افتح الرابط المعروض من المتصفح/الهاتف.
echo ============================================================
goto :eof
:err
echo.
echo [فشل] راجع الرسالة أعلاه — الأغلب checkpoint مفقود أو مكتبة غير مثبتة.
pause
exit /b 1
