from setuptools import setup, Extension
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

def readme():
    with open(os.path.join(dir_path, 'README.md')) as f:
        return f.read()

print(dir_path)

try:
    from Cython.Build import cythonize
    cython_function_path = os.path.join(dir_path, "pymskt/cython/cython_functions.pyx")
    print(cython_function_path)
    ext_modules = cythonize([Extension(name="pymskt.cython_functions",
                                       sources=[cython_function_path])],
                            annotate=True)
except ImportError:
    print('Import Error - Not building cython function!')
    ext_modules = None

setup(name='pymskt',
    version='0.0.1',
    description='vtk helper tools/functions for musculoskeletal analyses',
    long_description=readme(),
    url='https://github.com/gattia/pymskt.git',
    author='Anthony Gatti',
    author_email='anthony@neuralseg.com',
    license='proprietary - secret',
    ext_modules=ext_modules,
    packages=['pymskt',
              'pymskt.image',
              'pymskt.mesh',
              'pymskt.utils'],
    zip_safe=False,
    # tests_requires=['pytest'],
    # setup_requires=['pytest-runner', 'flake8']
    )
