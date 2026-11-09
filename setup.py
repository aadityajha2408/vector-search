"""Build the C++ extension in place:   python setup.py build_ext --inplace

Produces src/vecsearch_cpp.*.so (Linux/macOS) or src/vecsearch_cpp.*.pyd (Windows).
On Windows the simplest working setup is MSVC ("Build Tools for Visual Studio", C++ workload).
"""
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

if sys.platform == "win32":
    compile_args = ["/O2", "/arch:AVX2", "/openmp"]
    link_args = []
elif sys.platform == "darwin":
    compile_args = ["-O3", "-march=native"]  # Apple clang has no OpenMP by default
    link_args = []
else:
    compile_args = ["-O3", "-march=native", "-fopenmp"]
    link_args = ["-fopenmp"]

ext = Pybind11Extension(
    "src.vecsearch_cpp",
    ["cpp/bindings.cpp"],
    include_dirs=["cpp"],
    cxx_std=17,
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

setup(name="vecsearch", version="0.1", ext_modules=[ext], cmdclass={"build_ext": build_ext}, packages=[])
