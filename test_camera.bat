@echo off
echo ========================================
echo    Camera Test and Flask Application
echo ========================================
echo.

echo 🔍 Testing camera first...
python test_camera_simple.py

echo.
echo ========================================
echo    Starting Flask Application
echo ========================================
echo.

echo 🚀 Starting i-sight Flask application...
echo 📱 Open your browser to: http://localhost:5000
echo ⏹️  Press Ctrl+C to stop the application
echo.

python i_sight_flask_integrated.py

pause 