from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os
import sysconfig

cwd = os.path.abspath(os.path.dirname(__file__))
if not os.path.exists(os.path.join(cwd, 'PKG-INFO')):
    try:
        from Cython.Build import cythonize
        USE_CYTHON = True
    except ImportError:
        USE_CYTHON = False
else:
    USE_CYTHON = False

extra_compile_args = {
    'msvc': ['/std:c++14', "/openmp"],
    'unix': ['-std=c++11', "-fopenmp", "-Ofast", "-march=native"],
}

extra_link_args = {
    'msvc': ["/openmp"],
    'unix': ["-fopenmp"],
}


class build_ext_compiler_check(build_ext):
    def build_extensions(self):
        compiler = self.compiler.compiler_type

        try:
            compiler_name = os.path.basename(sysconfig.get_config_var("CC"))
        except TypeError:
            compiler_name = "None"

        if "clang" in compiler_name:
            compile_args = ["-std=c++11", "-Xpreprocessor -fopenmp", "-Ofast", "-march=native"]
            link_args = ['-lomp']
        else:
            compile_args = extra_compile_args[compiler]
            link_args = extra_link_args[compiler]

        for ext in self.extensions:
            ext.extra_compile_args = compile_args

        for ext in self.extensions:
            ext.extra_link_args = link_args

        build_ext.build_extensions(self)

    def run(self):
        import numpy
        import dimod
        self.include_dirs.append(numpy.get_include())
        self.include_dirs.append(dimod.get_include())
        build_ext.run(self)


if USE_CYTHON:
    ext = '.pyx'
else:
    ext = '.cpp'


extensions = [
    Extension(
        name='cvrp.tsp',
        sources=['./cvrp/swap_heuristic' + ext],
        include_dirs=["./cvrp/"],
        language='c++',
    )
]


if USE_CYTHON:
    print("using cython")
    extensions = cythonize(extensions, language='c++', annotate=False)

packages = find_packages()

setup(
    name='MVRP',
    packages=packages,
    ext_modules=extensions,
    cmdclass={'build_ext': build_ext_compiler_check},
    python_requires='>=3.7',
    zip_safe=False
)
