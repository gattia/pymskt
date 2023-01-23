from setuptools import setup, Extension
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

print(dir_path)

try:
    from Cython.Build import cythonize
    cython_function_path = "pymskt/cython/cython_functions.pyx"
    print(cython_function_path)
    ext_modules = cythonize([Extension(name="pymskt.cython_functions",
                                       sources=[cython_function_path])],
                            annotate=True)
except ImportError:
    print('Import Error - Not building cython function!')
    ext_modules = None

setup(ext_modules=ext_modules, zip_safe=False)
