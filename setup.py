import os
import sys

from setuptools import Extension, find_packages, setup

# Ensure we're running in the correct directory
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
print("Directory:", dir_path)

# Try to build the Cython extension if possible
try:
    from Cython.Build import cythonize

    cython_function_path = "pymskt/cython/cython_functions.pyx"
    print("Attempting to build Cython extension from:", cython_function_path)
    ext_modules = cythonize(
        [Extension(name="pymskt.cython_functions", sources=[cython_function_path])], annotate=True
    )
except Exception as e:
    print("WARNING: Failed to build Cython extension due to error:", e)
    print("Proceeding without compiled Cython modules. Performance may be affected.")
    ext_modules = []

# Optional build_ext command to allow installation to continue even if extension build fails.
from setuptools.command.build_ext import build_ext


class optional_build_ext(build_ext):
    def run(self):
        try:
            build_ext.run(self)
        except Exception as e:
            sys.stderr.write("WARNING: Building C extension failed, continuing without it.\n")

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except Exception as e:
            sys.stderr.write(f"WARNING: Failed to build extension {ext.name}, skipping it.\n")


# Setup without redundant metadata. Name, version, etc. are handled by pyproject.toml.
setup(
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": optional_build_ext},
    zip_safe=False,
)
