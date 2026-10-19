@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create .venv and install requirements.txt first. See README.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py
pause
