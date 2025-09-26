# Troubleshooting Guide - Animal Type Classification System

## Common Issues and Solutions

### 1. Backend Server Issues

#### Problem: "Connection error" or "Backend API is not running"
**Symptoms:**
- Frontend shows "Backend API is not running" error
- Can't connect to http://localhost:8000

**Solutions:**
1. Check if backend is running:
   ```cmd
   # Run in ATC directory
   start_backend.bat
   ```

2. Verify Python environment:
   ```cmd
   venv\Scripts\activate
   python --version
   ```

3. Check port availability:
   ```cmd
   netstat -an | findstr :8000
   ```

4. Try alternative port (edit backend/main.py):
   ```python
   uvicorn.run("main:app", host="0.0.0.0", port=8001)
   ```

#### Problem: "Module not found" errors
**Symptoms:**
- ImportError when starting backend
- Missing package errors

**Solutions:**
1. Reinstall requirements:
   ```cmd
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Update pip:
   ```cmd
   python -m pip install --upgrade pip
   ```

3. Check virtual environment activation:
   ```cmd
   where python
   # Should show path to venv\Scripts\python.exe
   ```

### 2. Frontend Issues

#### Problem: Streamlit won't start
**Symptoms:**
- "Command not found" streamlit
- Browser doesn't open

**Solutions:**
1. Ensure Streamlit is installed:
   ```cmd
   venv\Scripts\activate
   pip install streamlit
   ```

2. Start manually:
   ```cmd
   cd frontend
   streamlit run app.py
   ```

3. Check for port conflicts:
   ```cmd
   netstat -an | findstr :8501
   ```

#### Problem: "Connection error" in frontend
**Symptoms:**
- Can upload images but get connection errors
- API health check fails

**Solutions:**
1. Verify backend is running on correct port
2. Check CORS settings in backend/main.py
3. Try refreshing the browser page
4. Clear browser cache

### 3. Image Processing Issues

#### Problem: "Could not extract features from image"
**Symptoms:**
- Image uploads but feature extraction fails
- Classification returns empty results

**Solutions:**
1. Check image quality:
   - Use clear, well-lit images
   - Ensure animal is fully visible
   - Try different image formats (JPG, PNG)

2. Image requirements:
   - Minimum size: 300x300 pixels
   - Maximum size: 5MB
   - Clear side view of animal

3. Try with sample images:
   - Search online for "cow side view" or "buffalo side view"
   - Ensure animal is the main subject

#### Problem: Poor classification accuracy
**Symptoms:**
- Unrealistic scores
- Inconsistent classifications

**Solutions:**
1. Check image quality guidelines
2. Ensure correct species selection
3. Use images with good contrast
4. Avoid images with multiple animals

### 4. Database Issues

#### Problem: "Database connection failed"
**Symptoms:**
- History page doesn't load
- Classifications aren't saved

**Solutions:**
1. Check data directory permissions:
   ```cmd
   # Ensure data folder exists and is writable
   mkdir data
   ```

2. Delete and recreate database:
   ```cmd
   del data\atc_database.db
   # Restart backend to recreate
   ```

3. Check SQLite installation:
   ```cmd
   python -c "import sqlite3; print('SQLite OK')"
   ```

### 5. Model Training Issues

#### Problem: "Model not found" or training fails
**Symptoms:**
- Classification endpoint returns model errors
- Backend logs show training failures

**Solutions:**
1. Manual model training:
   ```python
   # In Python console from backend directory
   from models.classifier import AnimalClassifier
   classifier = AnimalClassifier()
   classifier.train_model(retrain=True)
   ```

2. Check data directory:
   ```cmd
   # Ensure data directory exists
   mkdir data
   ```

3. Memory issues:
   - Reduce training data size in classifier.py
   - Close other applications

### 6. Performance Issues

#### Problem: Slow image processing
**Symptoms:**
- Long wait times for classification
- Browser timeouts

**Solutions:**
1. Reduce image size before upload
2. Close unnecessary applications
3. Check system resources:
   ```cmd
   taskmgr
   # Monitor CPU and memory usage
   ```

4. Optimize OpenCV settings in image_processor.py

### 7. Setup Issues

#### Problem: Virtual environment creation fails
**Symptoms:**
- setup.bat fails
- Permission errors

**Solutions:**
1. Run as administrator:
   - Right-click setup.bat → "Run as administrator"

2. Manual setup:
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Check Python installation:
   ```cmd
   python --version
   # Should show Python 3.8 or higher
   ```

#### Problem: Package installation fails
**Symptoms:**
- pip install errors
- Missing dependencies

**Solutions:**
1. Update pip:
   ```cmd
   python -m pip install --upgrade pip
   ```

2. Install packages individually:
   ```cmd
   pip install fastapi uvicorn streamlit opencv-python scikit-learn
   ```

3. Use alternative package sources:
   ```cmd
   pip install -i https://pypi.org/simple/ package_name
   ```

### 8. Browser Issues

#### Problem: Frontend doesn't display correctly
**Symptoms:**
- Broken layout
- Missing elements

**Solutions:**
1. Try different browsers (Chrome, Firefox, Edge)
2. Disable browser extensions
3. Clear browser cache and cookies
4. Check browser console for JavaScript errors (F12)

#### Problem: File upload doesn't work
**Symptoms:**
- Upload button unresponsive
- File selection fails

**Solutions:**
1. Check file size (max 10MB)
2. Use supported formats (JPG, PNG, JPEG)
3. Try different browser
4. Check browser permissions for file access

## System Requirements

### Minimum Requirements
- **OS**: Windows 10 or later
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Internet**: Required for initial setup only

### Recommended Requirements
- **OS**: Windows 10/11
- **Python**: 3.9 or 3.10
- **RAM**: 8GB or more
- **Storage**: 5GB free space
- **CPU**: Multi-core processor for faster processing

## Log Files and Debugging

### Backend Logs
- Console output when running `python main.py`
- Check for error messages and stack traces

### Frontend Logs
- Streamlit logs in terminal
- Browser console (F12 → Console tab)

### Common Log Messages
1. **"Model loaded successfully"** - Good, classifier is ready
2. **"Database initialized"** - Good, database is working
3. **"Error preprocessing image"** - Image quality issue
4. **"Connection refused"** - Backend not running
5. **"Module not found"** - Missing package installation

## Getting Help

### Before Asking for Help
1. Check this troubleshooting guide
2. Verify system requirements
3. Try basic solutions (restart, reinstall)
4. Note exact error messages

### Information to Include
- Operating system version
- Python version (`python --version`)
- Exact error message
- Steps to reproduce the issue
- Screenshot if relevant

### Contact Information
- Create an issue on the project repository
- Include system information and error logs
- Describe steps taken to resolve the issue