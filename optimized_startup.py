#!/usr/bin/env python3
"""
Optimized Startup Script for Computer Vision Detection System
Handles proper process management and quit button functionality
"""

import subprocess
import time
import signal
import sys
import os
import threading
import requests
from datetime import datetime

class OptimizedSystemLauncher:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.detection_process = None
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.stop_system()
        sys.exit(0)
    
    def check_dependencies(self):
        """Check if required packages are installed"""
        required_packages = ['flask', 'opencv-python', 'numpy', 'requests']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages: {', '.join(missing_packages)}")
            print("Please install with: pip install -r requirements_backend_api.txt")
            return False
        
        return True
    
    def start_backend(self):
        """Start the backend API server"""
        try:
            print("🚀 Starting Backend API Server...")
            self.backend_process = subprocess.Popen(
                [sys.executable, 'backend_api.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for backend to start
            time.sleep(2)
            
            # Test backend health
            try:
                response = requests.get('http://localhost:8000/health', timeout=5)
                if response.status_code == 200:
                    print("✅ Backend API server started successfully")
                    return True
                else:
                    print("❌ Backend health check failed")
                    return False
            except requests.exceptions.RequestException:
                print("❌ Backend not responding")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False
    
    def start_frontend(self):
        """Start the Flask frontend"""
        try:
            print("🌐 Starting Flask Frontend...")
            self.frontend_process = subprocess.Popen(
                [sys.executable, 'flask_frontend.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for frontend to start
            time.sleep(3)
            
            # Test frontend
            try:
                response = requests.get('http://localhost:5000/api/status', timeout=5)
                if response.status_code == 200:
                    print("✅ Flask frontend started successfully")
                    return True
                else:
                    print("❌ Frontend health check failed")
                    return False
            except requests.exceptions.RequestException:
                print("❌ Frontend not responding")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start frontend: {e}")
            return False
    
    def wait_for_backend(self, timeout=30):
        """Wait for backend to be ready"""
        print("⏳ Waiting for backend to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get('http://localhost:8000/health', timeout=2)
                if response.status_code == 200:
                    print("✅ Backend is ready")
                    return True
            except:
                pass
            time.sleep(1)
        
        print("❌ Backend timeout")
        return False
    
    def wait_for_frontend(self, timeout=30):
        """Wait for frontend to be ready"""
        print("⏳ Waiting for frontend to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get('http://localhost:5000/api/status', timeout=2)
                if response.status_code == 200:
                    print("✅ Frontend is ready")
                    return True
            except:
                pass
            time.sleep(1)
        
        print("❌ Frontend timeout")
        return False
    
    def start_system(self):
        """Start the complete system"""
        print("🚦 Computer Vision Detection System - Optimized")
        print("=" * 60)
        
        # Check dependencies
        if not self.check_dependencies():
            print("❌ Dependencies check failed. Please install required packages.")
            return False
        
        # Start backend
        if not self.start_backend():
            return False
        
        # Wait for backend
        if not self.wait_for_backend():
            self.stop_system()
            return False
        
        # Start frontend
        if not self.start_frontend():
            self.stop_system()
            return False
        
        # Wait for frontend
        if not self.wait_for_frontend():
            self.stop_system()
            return False
        
        self.running = True
        
        print("\n🎉 System started successfully!")
        print("📡 Backend API: http://localhost:8000")
        print("🌐 Frontend: http://localhost:5000")
        print("\n🎮 Controls:")
        print("- Press Ctrl+C to stop the system")
        print("- Use the web interface to start/stop detection")
        print("- Quit button in web interface will properly stop detection")
        
        return True
    
    def stop_system(self):
        """Stop the complete system"""
        print("\n🛑 Stopping system...")
        
        # Stop detection first
        try:
            requests.post('http://localhost:8000/stop-detection', timeout=5)
            print("✅ Detection stopped")
        except:
            pass
        
        # Stop frontend
        if self.frontend_process:
            print("🛑 Stopping frontend...")
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
        
        # Stop backend
        if self.backend_process:
            print("🛑 Stopping backend...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
        
        self.running = False
        print("✅ System stopped")
    
    def run(self):
        """Run the system launcher"""
        if self.start_system():
            try:
                # Keep the system running
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Interrupted by user")
            finally:
                self.stop_system()

def main():
    launcher = OptimizedSystemLauncher()
    launcher.run()

if __name__ == "__main__":
    main()