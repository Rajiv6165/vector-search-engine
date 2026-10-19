import os
import sys
from setuptools import setup, Extension, find_packages

# Determine compiler arguments
extra_compile_args = []
extra_link_args = []
if sys.platform == "win32":
    extra_compile_args = ["/O2", "/fp:fast", "/std:c++14", "/openmp"]
    extra_link_args = []
else:
    extra_compile_args = ["-O3", "-ffast-math", "-std=c++11", "-fopenmp"]
    extra_link_args = ["-fopenmp"]

ext_modules = []
build_native = True

try:
    import numpy as np
    from Cython.Build import cythonize
    
    ext_modules = cythonize(
        [
            Extension(
                "vsearch._hnsw_native",
                sources=["src/vsearch/_hnsw_native.pyx"],
                include_dirs=[np.get_include(), "src/vsearch"],
                language="c++",
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
            )
        ],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    )
except Exception as e:
    print(f"Warning: Failed to configure native Cython extension ({e}).")
    print("Falling back to pure-Python implementation.")
    ext_modules = []

setup(
    ext_modules=ext_modules,
)
