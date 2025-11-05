# Local Dependency Verification Guide

This guide helps you verify that all dependencies are correctly installed and can be imported.

## Quick Start

### Windows
```bash
# Double-click or run in Command Prompt:
verify_local.bat

# Or manually:
python verify_dependencies.py
python test_imports.py
```

### Linux/Mac
```bash
# Make script executable (first time only):
chmod +x verify_local.sh

# Run verification:
./verify_local.sh

# Or manually:
python3 verify_dependencies.py
python3 test_imports.py
```

## What Each Script Does

### 1. `verify_dependencies.py`
- Checks if all packages from `requirements.txt` can be imported
- Displays installed versions vs expected versions
- Provides summary of missing packages

### 2. `test_imports.py`
- Tests actual imports from your project modules
- Verifies that dependencies work together
- Catches import errors early

### 3. `verify_local.bat` / `verify_local.sh`
- Complete automation script
- Installs dependencies if needed
- Runs both verification scripts
- Shows comprehensive results

## Manual Verification Steps

If you prefer to verify manually:

### Step 1: Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r Configuration/requirements.txt
```

### Step 2: Run Verification
```bash
# Check if packages are installed
python verify_dependencies.py

# Test if imports work
python test_imports.py
```

### Step 3: Test Your Application
```bash
# Try running your main script
python main.py --help

# Or test specific modules
python -c "from utils.ocr.ocr_utils import extract_lines_with_boxes; print('Import successful!')"
```

## Expected Output

### Successful Verification
```
============================================================
DEPENDENCY VERIFICATION SCRIPT
============================================================

Python Version: 3.11.x
Python Executable: /path/to/python

Step 1: Checking requirements.txt file...
✅ Found Configuration/requirements.txt
   Contains 54 package definitions

Step 2: Checking required packages...
----------------------------------------------------------------------
✅ cv2                  (package: opencv-python      )
    Installed: 4.12.0.88
    Expected:  4.12.0.88

✅ numpy                (package: numpy               )
    Installed: 2.2.6
    Expected:  2.2.6

... (more packages)

============================================================
SUMMARY
============================================================
Required packages: 12/12 installed

✅ ALL REQUIRED PACKAGES ARE INSTALLED!
```

### If Packages Are Missing
```
❌ SOME PACKAGES ARE MISSING!

To install missing packages, run:
  pip install -r Configuration/requirements.txt
```

## Troubleshooting

### Issue: `ModuleNotFoundError`
**Solution**: Install the missing package
```bash
pip install <package-name>
```

### Issue: Version Mismatch
**Solution**: Install the correct version
```bash
pip install <package-name>==<version>
```

### Issue: Import Errors in Project Modules
**Possible causes**:
1. Missing dependencies
2. Incorrect Python path
3. Module structure issues

**Solution**: Run with verbose mode
```bash
python test_imports.py --verbose
```

### Issue: Permission Errors (Linux/Mac)
**Solution**: Use `pip install --user` or use virtual environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r Configuration/requirements.txt
```

## Verification Checklist

- [ ] All core dependencies installed (12 packages)
- [ ] All project modules can be imported
- [ ] No import errors
- [ ] Version compatibility confirmed
- [ ] Application can start without errors

## Next Steps

After successful verification:
1. ✅ Dependencies are ready
2. ✅ You can run your application
3. ✅ Ready for deployment

If verification fails:
1. Check error messages
2. Install missing packages
3. Fix any import issues
4. Re-run verification

## Additional Notes

- **Virtual Environment**: Highly recommended to avoid conflicts
- **Version Pinning**: requirements.txt uses exact versions for reproducibility
- **Docker**: If using Docker, dependencies are installed during build
- **CI/CD**: These scripts can be integrated into CI/CD pipelines

