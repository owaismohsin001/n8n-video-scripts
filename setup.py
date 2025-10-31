from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "video_translate_cython",
        ["video_translate_cython.pyx"],
        extra_compile_args=["/openmp"],  # for MSVC compiler
        extra_link_args=[],
    )
]


setup(
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"})
)
