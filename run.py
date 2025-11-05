#!/usr/bin/env python3
"""
Video Translation Script - Main Entry Point
Supports both Cython-optimized and pure Python implementations
"""

import argparse
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import importlib.util

# Force-load the compiled Cython module manually if normal import fails
try:
    import video_translate_cython as vt
except ImportError:
    pyd_path = Path(__file__).with_name("video_translate_cython.cp311-win_amd64.pyd")
    if pyd_path.exists():
        spec = importlib.util.spec_from_file_location("video_translate_cython", pyd_path)
        vt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vt)
        sys.modules["video_translate_cython"] = vt
    else:
        print("❌ Still can’t find the compiled module:", pyd_path)
        sys.exit(1)

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_banner():
    """Print application banner"""
    banner = f"""
{Colors.HEADER}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════╗
║         VIDEO TRANSLATION WITH OCR                        ║
║         High-Performance Parallel Processing              ║
╚═══════════════════════════════════════════════════════════╝
{Colors.ENDC}
"""
    print(banner)


def check_file_exists(filepath, file_type="File"):
    """Check if a file exists and is accessible"""
    if not os.path.exists(filepath):
        print(f"{Colors.FAIL}✗ {file_type} not found: {filepath}{Colors.ENDC}")
        return False
    if not os.path.isfile(filepath):
        print(f"{Colors.FAIL}✗ {file_type} is not a file: {filepath}{Colors.ENDC}")
        return False
    return True


def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []
    
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    if missing:
        print(f"{Colors.FAIL}✗ Missing dependencies: {', '.join(missing)}{Colors.ENDC}")
        print(f"{Colors.WARNING}Install with: pip install {' '.join(missing)}{Colors.ENDC}")
        return False
    
    return True


def load_translator_module():
    """
    Try to load Cython-optimized module first, fallback to Python version
    Returns: (module, is_cython)
    """
    # Try Cython version first (video_translate_cython)
    try:
        import video_translate_cython as translator
        # Check if it's actually the compiled Cython module
        if hasattr(translator, '__file__') and translator.__file__.endswith(('.so', '.pyd', '.dll')):
            print(f"{Colors.OKGREEN}✓ Using Cython-optimized module (video_translate_cython){Colors.ENDC}")
            return translator, True
    except ImportError:
        pass
    
    # Try alternative name (video_translator)
    try:
        import video_translator as translator
        if hasattr(translator, '__file__') and translator.__file__.endswith(('.so', '.pyd', '.dll')):
            print(f"{Colors.OKGREEN}✓ Using Cython-optimized module (video_translator){Colors.ENDC}")
            return translator, True
    except ImportError:
        pass
    
    # Try Python version
    try:
        # If original file exists with different name
        import video_translator_py as translator
        print(f"{Colors.WARNING}⚠ Using Python version (Cython not available){Colors.ENDC}")
        print(f"{Colors.WARNING}  Build Cython for better performance: python setup.py build_ext --inplace{Colors.ENDC}")
        return translator, False
    except ImportError:
        pass
    
    # Last resort: try importing original from current directory
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        # Try to import the original Python file
        for module_name in ['video_translate', 'translate_video', 'video_translator']:
            try:
                translator = __import__(module_name)
                print(f"{Colors.WARNING}⚠ Using standard Python module ({module_name}){Colors.ENDC}")
                return translator, False
            except ImportError:
                continue
    except Exception:
        pass
    
    print(f"{Colors.FAIL}✗ Cannot load video translator module!{Colors.ENDC}")
    print(f"{Colors.WARNING}  Make sure one of these exists:{Colors.ENDC}")
    print(f"{Colors.WARNING}  - video_translate_cython.pyd/.so (Cython compiled){Colors.ENDC}")
    print(f"{Colors.WARNING}  - video_translator_cython.py (Python version){Colors.ENDC}")
    return None, False


def validate_color(color_str):
    """Validate color input"""
    valid_colors = [
        'black', 'white', 'red', 'green', 'blue', 'yellow', 
        'cyan', 'magenta', 'orange', 'purple', 'pink', 'brown', 'gray'
    ]
    
    color_lower = color_str.lower()
    
    # Check if it's a hex color
    if color_str.startswith('#'):
        if len(color_str) in [4, 7] and all(c in '0123456789ABCDEFabcdef' for c in color_str[1:]):
            return color_str
        else:
            print(f"{Colors.WARNING}⚠ Invalid hex color: {color_str}{Colors.ENDC}")
            return None
    
    # Check if it's a named color
    if color_lower in valid_colors:
        return color_lower
    
    print(f"{Colors.WARNING}⚠ Invalid color: {color_str}{Colors.ENDC}")
    print(f"{Colors.WARNING}  Valid colors: {', '.join(valid_colors)}{Colors.ENDC}")
    print(f"{Colors.WARNING}  Or use hex format: #RRGGBB{Colors.ENDC}")
    return None


def create_output_directory(output_path):
    """Create output directory if it doesn't exist"""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"{Colors.OKGREEN}✓ Created output directory: {output_dir}{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}✗ Failed to create output directory: {e}{Colors.ENDC}")
            return False
    return True


def estimate_processing_time(video_path):
    """Estimate processing time based on video properties"""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        duration = frame_count / fps if fps > 0 else 0
        
        # Rough estimation: 0.5-2 seconds per frame depending on complexity
        # This varies greatly based on OCR complexity and hardware
        estimated_seconds = frame_count * 1.0  # Conservative estimate
        estimated_minutes = estimated_seconds / 60
        
        print(f"\n{Colors.OKCYAN}📊 Video Information:{Colors.ENDC}")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps:.2f}")
        print(f"   Frames: {frame_count}")
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   {Colors.WARNING}Estimated processing time: {estimated_minutes:.1f} - {estimated_minutes*2:.1f} minutes{Colors.ENDC}")
        print(f"   (Actual time depends on text complexity and hardware)")
        
    except Exception as e:
        print(f"{Colors.WARNING}⚠ Could not estimate processing time: {e}{Colors.ENDC}")


def main():
    """Main entry point"""
    print_banner()
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Overlay translations on video frames with parallel processing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run.py --video input.mp4 --targetLang English
  
  # Full options
  python run.py --video input.mp4 --font font.ttf --fontSize 24 \\
                --out output.mp4 --targetLang English --fontColor Red \\
                --sourceLang chinese --parallel
  
  # Sequential processing (legacy mode)
  python run.py --video input.mp4 --sequential
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--video",
        dest="video_path",
        required=True,
        help="Path to input video file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--font",
        dest="font_path",
        default=None,
        help="Path to TTF font file (optional, uses default if not specified)"
    )
    
    parser.add_argument(
        "--fontSize",
        dest="font_size",
        type=int,
        default=35,
        help="Font size for overlay text (default: 35)"
    )
    
    parser.add_argument(
        "--out",
        dest="out_path",
        default=None,
        help="Output video path (default: output/translated.mp4)"
    )
    
    parser.add_argument(
        "--targetLang",
        dest="target_language",
        default="English",
        help="Target language for translation (default: English)"
    )
    
    parser.add_argument(
        "--fontColor",
        dest="font_color",
        default="black",
        help="Font color for translation overlay (default: black)"
    )
    
    parser.add_argument(
        "--sourceLang",
        dest="source_language",
        default="english",
        help="Source language of the video (default: english)"
    )
    
    parser.add_argument(
        "--parallel",
        dest="use_parallel",
        action="store_true",
        default=True,
        help="Use parallel batch processing (default: True)"
    )
    
    parser.add_argument(
        "--sequential",
        dest="use_parallel",
        action="store_false",
        help="Use sequential processing (legacy mode, slower)"
    )
    
    parser.add_argument(
        "--skipChecks",
        dest="skip_checks",
        action="store_true",
        help="Skip dependency and file validation checks"
    )
    
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print(f"{Colors.OKBLUE}⚙️  Configuration:{Colors.ENDC}")
    print(f"   Video: {args.video_path}")
    print(f"   Font: {args.font_path or 'Default'}")
    print(f"   Font Size: {args.font_size}")
    print(f"   Font Color: {args.font_color}")
    print(f"   Source Language: {args.source_language}")
    print(f"   Target Language: {args.target_language}")
    print(f"   Processing Mode: {'Parallel' if args.use_parallel else 'Sequential'}")
    print(f"   Output: {args.out_path or 'output/translated.mp4'}")
    print()
    
    # Validation phase
    if not args.skip_checks:
        print(f"{Colors.OKBLUE}🔍 Validating inputs...{Colors.ENDC}")
        
        # Check dependencies
        if not check_dependencies():
            return 1
        
        # Check input video
        if not check_file_exists(args.video_path, "Input video"):
            return 1
        
        # Check font file if specified
        if args.font_path and not check_file_exists(args.font_path, "Font file"):
            return 1
        
        # Validate color
        validated_color = validate_color(args.font_color)
        if validated_color is None:
            return 1
        args.font_color = validated_color
        
        print(f"{Colors.OKGREEN}✓ All validations passed{Colors.ENDC}\n")
    
    # Set default output path
    if args.out_path is None:
        from constants.paths import OUTPUT_PATH
        args.out_path = OUTPUT_PATH
    
    # Create output directory
    if not create_output_directory(args.out_path):
        return 1
    
    # Load translator module
    translator, is_cython = load_translator_module()
    if translator is None:
        return 1
    
    # Show processing estimate
    estimate_processing_time(args.video_path)
    
    # Confirm before starting
    # print(f"\n{Colors.WARNING}Press Enter to start processing (Ctrl+C to cancel)...{Colors.ENDC}")
    # try:
    #     input()
    # except KeyboardInterrupt:
    #     print(f"\n{Colors.WARNING}Cancelled by user{Colors.ENDC}")
    #     return 0
    
    # Start processing
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}STARTING VIDEO PROCESSING{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")
    
    start_time = time.time()
    
    try:
        # Call the main function
        translator.function_overlaying_continuous(
            video_path=args.video_path,
            font_path=args.font_path,
            font_size=args.font_size,
            out_path=args.out_path,
            target_language=args.target_language,
            font_color=args.font_color,
            source_language=args.source_language,
            use_parallel=args.use_parallel
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        # Success message
        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}{Colors.BOLD}✅ PROCESSING COMPLETE!{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"\n{Colors.OKGREEN}📊 Summary:{Colors.ENDC}")
        print(f"   Processing time: {minutes}m {seconds}s")
        print(f"   Output saved to: {args.out_path}")
        print(f"   Implementation: {'Cython (Optimized)' if is_cython else 'Python'}")
        
        # Check output file
        if os.path.exists(args.out_path):
            file_size = os.path.getsize(args.out_path) / (1024 * 1024)  # MB
            print(f"   Output file size: {file_size:.2f} MB")
        
        print(f"\n{Colors.OKGREEN}🎉 Your translated video is ready!{Colors.ENDC}\n")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}⚠ Processing interrupted by user{Colors.ENDC}")
        return 130
        
    except Exception as e:
        print(f"\n\n{Colors.FAIL}{'='*60}{Colors.ENDC}")
        print(f"{Colors.FAIL}❌ ERROR OCCURRED{Colors.ENDC}")
        print(f"{Colors.FAIL}{'='*60}{Colors.ENDC}")
        print(f"{Colors.FAIL}Error: {str(e)}{Colors.ENDC}")
        
        if args.verbose:
            import traceback
            print(f"\n{Colors.FAIL}Full traceback:{Colors.ENDC}")
            traceback.print_exc()
        else:
            print(f"\n{Colors.WARNING}Run with --verbose for full error details{Colors.ENDC}")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())