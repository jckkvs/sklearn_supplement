from distutils.core import setup
from distutils.extension import Extension
from pathlib import Path

import numpy as np
from setuptools import find_packages, setup

version = "0.1.20"

try:
    from Cython.Distutils import build_ext

    "sklearn_expansion\manifold/_utils.pyx"
    sourcefiles = [Path("sklearn_expansion") / "manifold" / "_utils.pyx"]
    sourcefiles = ["_utils.pyx"]

    setup(
        name="sklearn_expansion",
        version=version,
        description="sklearnの拡張ライブラリ",
        author="Yuki Horie",
        author_email="Yuki.Horie@mitsuichemicals.com",
        packages=find_packages(),
        cmdclass={"build_ext": build_ext},
        include_dirs=[np.get_include()],
        ext_modules=[Extension("qsne_utils", sourcefiles)],
        install_requires=[
            "cvxpy",
            "scikit-learn",
            "more-itertools",
            "numpy",
            "pandas",
            "optuna",
            "deap",
            "boruta",
        ],
    )

except:
    setup(
        name="sklearn_expansion",
        version=version,
        description="sklearnの拡張ライブラリ",
        author="Yuki Horie",
        author_email="Yuki.Horie@mitsuichemicals.com",
        packages=find_packages(),
        install_requires=[
            "scikit-learn",
            "numpy",
            "pandas",
            "optuna",
            "deap",
            "boruta",
        ],
    )
