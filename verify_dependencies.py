#!/usr/bin/env python3
"""
Dependency Verification Script
Tests if all required libraries can be imported and checks their versions.
"""

import sys
import importlib
from typing import Dict, Tuple, List

# Define all required packages and their expected versions from requirements.txt
REQUIRED_PACKAGES = {
    'PIL': ('pillow', '11.3.0'),
    'cv2': ('opencv-python', '4.12.0.88'),
    'numpy': ('numpy', '2.2.6'),
    'pandas': ('pandas', '2.3.2'),
    'easyocr': ('easyocr', '1.7.2'),
    'pytesseract': ('pytesseract', '0.3.13'),
    'openai': ('openai', '1.109.1'),
    'dotenv': ('python-dotenv', '1.1.1'),
    'arabic_reshaper': ('arabic-reshaper', '3.0.0'),
    'bidi': ('python-bidi', '0.6.6'),
    'sympy': ('sympy', '1.14.0'),
}

# Packages that might not have version info easily accessible
OPTIONAL_PACKAGES = {
    'gdown': ('gdown', '5.2.0'),
}

# Standard library modules (no need to check)
STDLIB_MODULES = {
    'os', 'sys', 'time', 'json', 'datetime', 'collections', 'typing',
    'dataclasses', 're', 'argparse', 'difflib', 'pathlib', 'importlib',
    'multiprocessing', 'subprocess', 'concurrent.futures'
}

def get_package_version(package_name: str) -> str:
    """Get version of an installed package."""
    try:
        # Try common version attributes
        module = importlib.import_module(package_name)
        if hasattr(module, '__version__'):
            return module.__version__
        elif hasattr(module, 'version'):
            return str(module.version)
        elif hasattr(module, 'VERSION'):
            return str(module.VERSION)
        else:
            return "unknown (installed)"
    except Exception as e:
        return f"error: {str(e)}"

def check_import(module_name: str, package_info: Tuple[str, str]) -> Tuple[bool, str, str]:
    """
    Check if a module can be imported.
    Returns: (success, installed_version, expected_version)
    """
    package_name, expected_version = package_info
    
    try:
        module = importlib.import_module(module_name)
        installed_version = get_package_version(module_name)
        return True, installed_version, expected_version
    except ImportError as e:
        return False, f"NOT INSTALLED: {str(e)}", expected_version
    except Exception as e:
        return False, f"ERROR: {str(e)}", expected_version

def verify_requirements_file():
    """Verify that requirements.txt exists and can be read."""
    import os
    req_file = "Configuration/requirements.txt"
    
    if not os.path.exists(req_file):
        print(f"❌ ERROR: {req_file} not found!")
        return False
    
    print(f"✅ Found {req_file}")
    
    # Try to read it
    try:
        with open(req_file, 'r') as f:
            lines = f.readlines()
            print(f"   Contains {len(lines)} package definitions")
        return True
    except Exception as e:
        print(f"❌ ERROR: Could not read {req_file}: {e}")
        return False

def main():
    """Main verification function."""
    print("=" * 70)
    print("DEPENDENCY VERIFICATION SCRIPT")
    print("=" * 70)
    print()
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print()
    
    # Verify requirements.txt exists
    print("Step 1: Checking requirements.txt file...")
    if not verify_requirements_file():
        print("\n❌ Cannot proceed without requirements.txt")
        return 1
    print()
    
    # Check all required packages
    print("Step 2: Checking required packages...")
    print("-" * 70)
    
    results: List[Tuple[str, bool, str, str]] = []
    
    for module_name, package_info in REQUIRED_PACKAGES.items():
        success, installed_ver, expected_ver = check_import(module_name, package_info)
        results.append((module_name, success, installed_ver, expected_ver))
        
        status = "✅" if success else "❌"
        print(f"{status} {module_name:20s} (package: {package_info[0]:20s})")
        if success:
            print(f"    Installed: {installed_ver}")
            print(f"    Expected:  {expected_ver}")
        else:
            print(f"    Error:     {installed_ver}")
        print()
    
    # Check optional packages
    print("Step 3: Checking optional packages...")
    print("-" * 70)
    
    for module_name, package_info in OPTIONAL_PACKAGES.items():
        success, installed_ver, expected_ver = check_import(module_name, package_info)
        status = "✅" if success else "⚠️"
        print(f"{status} {module_name:20s} (package: {package_info[0]:20s})")
        if success:
            print(f"    Installed: {installed_ver}")
        else:
            print(f"    Status:    {installed_ver}")
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    successful = sum(1 for _, success, _, _ in results if success)
    total = len(results)
    
    print(f"Required packages: {successful}/{total} installed")
    print()
    
    if successful == total:
        print("✅ ALL REQUIRED PACKAGES ARE INSTALLED!")
        print()
        print("Next steps:")
        print("1. You can now run your application")
        print("2. Test imports in your code to ensure everything works")
        return 0
    else:
        print("❌ SOME PACKAGES ARE MISSING!")
        print()
        print("To install missing packages, run:")
        print("  pip install -r Configuration/requirements.txt")
        print()
        print("Or install in a virtual environment:")
        print("  python -m venv venv")
        print("  source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        print("  pip install -r Configuration/requirements.txt")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

