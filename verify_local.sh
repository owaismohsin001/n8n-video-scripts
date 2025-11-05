#!/bin/bash
# Local Dependency Verification Script for Linux/Mac

echo "============================================================"
echo "LOCAL DEPENDENCY VERIFICATION"
echo "============================================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and add it to your PATH"
    exit 1
fi

echo "Python found:"
python3 --version
echo

# Check if requirements.txt exists
if [ ! -f "Configuration/requirements.txt" ]; then
    echo "ERROR: Configuration/requirements.txt not found!"
    exit 1
fi

echo "Step 1: Installing/Updating dependencies..."
echo "--------------------------------------------"
pip3 install -r Configuration/requirements.txt
if [ $? -ne 0 ]; then
    echo
    echo "WARNING: Some packages may have failed to install"
    echo "Continuing with verification..."
    echo
else
    echo
    echo "Dependencies installed successfully!"
    echo
fi

echo "Step 2: Verifying dependencies..."
echo "--------------------------------------------"
python3 verify_dependencies.py
if [ $? -ne 0 ]; then
    echo
    echo "Some dependencies are missing. Please check the output above."
    exit 1
fi

echo
echo "Step 3: Testing project imports..."
echo "--------------------------------------------"
python3 test_imports.py
if [ $? -ne 0 ]; then
    echo
    echo "Some imports failed. Please check the output above."
    exit 1
fi

echo
echo "============================================================"
echo "VERIFICATION COMPLETE!"
echo "============================================================"
echo
echo "Your project dependencies are verified and ready to use."

