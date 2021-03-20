@echo off
echo I'll give you stocks
call "C:\Users\juanlazarde\Documents\Python Scripts\candlestick screener\venv\Scripts\activate.bat"
"C:\Users\juanlazarde\Documents\Python Scripts\candlestick screener\venv\Scripts\python.exe" "C:\Users\juanlazarde\Documents\Python Scripts\candlestick screener\app.py"
call "C:\Users\juanlazarde\Documents\Python Scripts\candlestick screener\venv\Scripts\deactivate.bat"
pause
exit