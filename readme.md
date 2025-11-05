venv\Scripts\activate.bat

activate on any terminal by running:
venv\Scripts\activate

pip install -r Configuration/requirements.txt

python main.py --video "input_videos/test_cut.mp4" --font "fonts/NotoSans-RegularEnglish.ttf" --fontSize 24 --out "output/translated.mp4" --targetLang "English" --fontColor "Red" --sourceLang "chinese" --parallel

python run.py --video "input_videos/test_cut.mp4" --font "fonts/NotoSans-RegularEnglish.ttf" --fontSize 24 --out "output/translated.mp4" --targetLang "English" --fontColor "Red" --sourceLang "chinese" --parallel

run setup to convert .pyx files to .c files, and window specific buil
python setup.py build_ext --inplace
