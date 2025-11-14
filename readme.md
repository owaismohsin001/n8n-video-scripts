python -m venv venv
source venv/Scripts/activate

## Windows Installation Notes

On Windows, you need to install `ninja-build` before installing Python dependencies:

**Option 1: Use the setup script (recommended)**

```bash
setup_windows.bat
venv\bin\python -m pip install -r Configuration/requirements.txt
```

**Option 2: Install ninja-build manually**

1. Download ninja-win.zip from: https://github.com/ninja-build/ninja/releases/download/v1.13.0/ninja-win.zip
2. Extract `ninja.exe` to a directory in your PATH (e.g., `C:\Windows\System32` or add to system PATH)
3. Or install via Chocolatey (requires admin): `choco install ninja -y`

**Option 3: Use the downloaded ninja.exe**
If you already have `ninja.exe` in the project root, add it to PATH:

```bash
set "PATH=%CD%;%PATH%"
venv\bin\python -m pip install -r Configuration/requirements.txt
```

## Install Python Dependencies

pip install -r Configuration/requirements.txt
venv\bin\python -m pip install -r Configuration/requirements.txt

python optimal_latest.py --video "input_videos/test_cut.mp4" --font "fonts/NotoSans-RegularEnglish.ttf" --fontSize 24 --out "output/translated.mp4" --targetLang "English" --fontColor "Red" --sourceLang "chinese" --parallel

venv\bin\python optimal_latest.py --video "input_videos/test_cut.mp4" --font "fonts/NotoSans-RegularEnglish.ttf" --fontSize 24 --out "output/translated.mp4" --targetLang "English" --fontColor "Red" --sourceLang "chinese" --parallel

python run.py --video "input_videos/test_cut.mp4" --font "fonts/NotoSans-RegularEnglish.ttf" --fontSize 24 --out "output/translated.mp4" --targetLang "English" --fontColor "Red" --sourceLang "chinese" --parallel

run setup to convert .pyx files to .c files, and window specific buil
python setup.py build_ext --inplace

python utils/ocr/ocr_utils.py
