@echo off
REM OCR Pipeline API Setup Script for Windows

echo.
echo 🚀 OCR Pipeline API - Setup Script (Windows)
echo ============================================

REM Check Python version
echo.
echo Checking Python version...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python 3.11+
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo ✓ Found: %PYTHON_VERSION%

REM Create virtual environment
echo.
echo Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo ✓ Virtual environment created
) else (
    echo ✓ Virtual environment already exists
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo ✓ Virtual environment activated

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
echo ✓ pip upgraded

REM Install dependencies
echo.
echo Installing dependencies...
if exist "pyproject.toml" (
    pip install -e .
) else if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    echo ❌ No pyproject.toml or requirements.txt found!
    exit /b 1
)
echo ✓ Dependencies installed

REM Setup environment file
echo.
echo Setting up environment...
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env
        echo ✓ Created .env from .env.example
    )
) else (
    echo ✓ .env already exists
)

REM Create model directories
echo.
echo Creating model directories...
if not exist "model_path\text_detection" mkdir model_path\text_detection
if not exist "model_path\text_recognition" mkdir model_path\text_recognition
echo ✓ Model directories created

echo.
echo ✅ Setup complete!
echo.
echo 📝 Next steps:
echo 1. Review and update .env file if needed
echo 2. Activate venv: .venv\Scripts\activate.bat
echo 3. Start server: python -m uvicorn app.main:app --reload
echo 4. Visit: http://localhost:8000/api/docs
echo.
echo 📚 For more information, see QUICKSTART.md or README.md
