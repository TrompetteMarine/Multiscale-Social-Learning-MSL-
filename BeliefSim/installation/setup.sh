#!/bin/bash
# setup.sh - Unix/Linux/macOS setup script for BeliefSim

echo "🌊 BeliefSim Setup Script"
echo "========================"
echo ""

# Check if Julia is installed
if ! command -v julia &> /dev/null; then
    echo "❌ Julia not found in PATH"
    echo ""
    echo "Please install Julia first:"
    echo "  1. Download Julia from https://julialang.org/downloads/"
    echo "  2. Add Julia to your PATH"
    echo "  3. Re-run this setup script"
    echo ""
    exit 1
fi

echo "✅ Julia found: $(julia --version)"
echo ""

# Run the Julia setup script
echo "🚀 Running Julia package installation..."
julia setup.jl

# Check if setup was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "Run verification:"
    echo "  julia install_verification.jl"
    echo ""
    echo "Start exploring:"
    echo "  julia demo_example.jl"
else
    echo ""
    echo "❌ Setup encountered errors"
    echo "Check the output above for details"
    exit 1
fi

---

REM setup.bat - Windows setup script for BeliefSim
@echo off
echo 🌊 BeliefSim Setup Script
echo ========================
echo.

REM Check if Julia is installed
julia --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Julia not found in PATH
    echo.
    echo Please install Julia first:
    echo   1. Download Julia from https://julialang.org/downloads/
    echo   2. Add Julia to your PATH
    echo   3. Re-run this setup script
    echo.
    pause
    exit /b 1
)

julia --version
echo.

REM Run the Julia setup script
echo 🚀 Running Julia package installation...
julia setup.jl

if %errorlevel% equ 0 (
    echo.
    echo 🎉 Setup completed successfully!
    echo.
    echo Run verification:
    echo   julia install_verification.jl
    echo.
    echo Start exploring:
    echo   julia demo_example.jl
    pause
) else (
    echo.
    echo ❌ Setup encountered errors
    echo Check the output above for details
    pause
    exit /b 1
)