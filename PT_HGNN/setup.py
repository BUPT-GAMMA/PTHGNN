from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name = "cython_util",
        sources = ["*.pyx"],
        extra_compile_args=["-std=c++11"])
]

setup(
    ext_modules=cythonize(extensions, language="c++"),
    include_dirs=[np.get_include()]
)