@echo off
echo ========================================
echo    i-sight Flask Integrated System
echo    with YOLO11n Object Detection
echo ========================================
echo.

echo 🔍 Checking Python installation...
python --version
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo.
echo 🔧 Setting up virtual environment...
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo 📦 Installing dependencies...
echo Installing Flask and core dependencies...
pip install -r requirements_i_sight_flask.txt

echo.
echo 🚀 Installing YOLO11n (ultralytics)...
pip install ultralytics

echo.
echo ✅ Setup complete!
echo.
echo 🚀 Starting i-sight Flask application...
echo 📱 Open your browser to: http://localhost:5000
echo 📹 Video Stream: http://localhost:5000/video-feed
echo 🔗 API Endpoints: http://localhost:5000/api/
echo.
echo ⏹️  Press Ctrl+C to stop the application
echo.

python i_sight_flask_integrated.py

pause 