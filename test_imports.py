#!/usr/bin/env python3
"""
Test actual imports from project files to verify all dependencies work together.
"""

import sys
import traceback

def test_import(module_path: str, description: str) -> bool:
    """Test importing a module and return True if successful."""
    try:
        if '/' in module_path or '\\' in module_path:
            # Handle file paths
            import importlib.util
            spec = importlib.util.spec_from_file_location(module_path.split('/')[-1].split('\\')[-1], module_path)
            if spec is None or spec.loader is None:
                print(f"❌ {description}: Could not create spec for {module_path}")
                return False
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            # Handle module names
            __import__(module_path)
        print(f"✅ {description}")
        return True
    except Exception as e:
        print(f"❌ {description}: {str(e)}")
        if '--verbose' in sys.argv or '-v' in sys.argv:
            traceback.print_exc()
        return False

def main():
    """Test all critical imports."""
    print("=" * 70)
    print("TESTING PROJECT IMPORTS")
    print("=" * 70)
    print()
    
    tests = [
        # Core dependencies
        ("cv2", "OpenCV (cv2)"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow (PIL)"),
        ("pandas", "Pandas"),
        ("easyocr", "EasyOCR"),
        ("pytesseract", "PyTesseract"),
        ("openai", "OpenAI"),
        ("dotenv", "python-dotenv"),
        ("arabic_reshaper", "Arabic Reshaper"),
        ("bidi.algorithm", "python-bidi"),
        ("sympy", "SymPy"),
        
        # Project modules
        ("audioUtils", "audioUtils module"),
        ("utils.vision", "utils.vision module"),
        ("utils.overlay_utils", "utils.overlay_utils module"),
        ("utils.translate_utils", "utils.translate_utils module"),
        ("utils.ocr.ocr_utils", "utils.ocr.ocr_utils module"),
        ("utils.pattern", "utils.pattern module"),
        ("utils.system_resources", "utils.system_resources module"),
        ("constants.index", "constants.index module"),
        ("constants.paths", "constants.paths module"),
        ("constants.ocr", "constants.ocr module"),
    ]
    
    print("Testing core dependencies...")
    print("-" * 70)
    core_results = []
    for module, desc in tests[:11]:  # First 11 are core dependencies
        result = test_import(module, desc)
        core_results.append(result)
    
    print()
    print("Testing project modules...")
    print("-" * 70)
    project_results = []
    for module, desc in tests[11:]:  # Rest are project modules
        result = test_import(module, desc)
        project_results.append(result)
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    core_passed = sum(core_results)
    core_total = len(core_results)
    project_passed = sum(project_results)
    project_total = len(project_results)
    
    print(f"Core dependencies: {core_passed}/{core_total} passed")
    print(f"Project modules: {project_passed}/{project_total} passed")
    print(f"Total: {core_passed + project_passed}/{core_total + project_total} passed")
    print()
    
    if core_passed == core_total and project_passed == project_total:
        print("✅ ALL TESTS PASSED! Your project is ready to run.")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print()
        if core_passed < core_total:
            print("Missing core dependencies. Install with:")
            print("  pip install -r Configuration/requirements.txt")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

