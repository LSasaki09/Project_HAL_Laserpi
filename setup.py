from setuptools import setup, Extension
from pybind11.setup_helpers import build_ext
import os
import sys

"""Setup script for building Python bindings for the HALaser libe1701.so library on Linux ARM64."""

try:
    import pybind11
except ImportError:
    print("Error: pybind11 is not installed. Run: pip install pybind11")
    sys.exit(1)

# Define paths
root_dir = os.path.abspath(".")
lib_path = os.path.join(root_dir, "libe1701")
include_path = lib_path

# Base name of the .so file without the "lib" prefix or extension
lib_file = "e1701"

# Checks
if not os.path.isfile(os.path.join(lib_path, "libe1701.so")):
    print("Error: libe1701.so not found")
    sys.exit(1)
if not os.path.isfile(os.path.join(lib_path, "libe1701.h")):
    print("Error: libe1701.h not found")
    sys.exit(1)

ext_modules = [
    Extension(
        "libe1701py",
        sources=["bindings/binding.cpp"],
        include_dirs=[
            pybind11.get_include(),
            include_path
        ],
        library_dirs=[lib_path],
        libraries=["e1701"],  # without the "lib" prefix or ".so"
        extra_compile_args=["-std=c++17"],
        language="c++"
    )
]

setup(
    name="libe1701py",
    version="0.1",
    description="Python bindings for HALaser libe1701.so (Linux ARM64)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
