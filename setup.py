import os
import shutil
from pathlib import Path

from pip._internal.req import parse_requirements
from setuptools import setup, find_namespace_packages
from setuptools.command.build_py import build_py

# TODO: find from extras maybe
HOST_MLIR_PYTHON_PACKAGE_PREFIX = "triton_mlir"
PACKAGE_NAME = f"{HOST_MLIR_PYTHON_PACKAGE_PREFIX.replace('.', '-').replace('_', '-')}-python-extras"


def load_requirements(fname):
    reqs = parse_requirements(fname, session="hack")
    return [str(ir.requirement) for ir in reqs]


packages = [
    f"{HOST_MLIR_PYTHON_PACKAGE_PREFIX}.extras.{p}"
    for p in find_namespace_packages(where="mlir/extras")
] + [f"{HOST_MLIR_PYTHON_PACKAGE_PREFIX}.extras"]


class MyInstallData(build_py):
    def run(self):
        build_py.run(self)
        try:
            from llvm import amdgcn

            build_dir = os.path.join(
                *([self.build_lib] + HOST_MLIR_PYTHON_PACKAGE_PREFIX.split("."))
            )
            shutil.copy(
                amdgcn.__file__,
                Path(build_dir) / "extras" / "dialects" / "ext" / "llvm" / "amdgcn.py",
            )
        except ImportError:
            pass


setup(
    name=PACKAGE_NAME,
    version="0.0.7",
    description="The missing pieces (as far as boilerplate reduction goes) of the upstream MLIR python bindings.",
    license="LICENSE",
    install_requires=load_requirements("requirements.txt"),
    extras_require={
        "test": ["pytest", "mlir-native-tools", "astpretty"],
        "torch-mlir": ["torch-mlir-core"],
        "jax": ["jax[cpu]"],
        "mlir": ["mlir-python-bindings"],
        "eudsl": ["eudsl-llvmpy"],
    },
    python_requires=">=3.8",
    packages=packages,
    # lhs is package namespace, rhs is path (relative to this setup.py)
    package_dir={
        f"{HOST_MLIR_PYTHON_PACKAGE_PREFIX}.extras": "mlir/extras",
    },
    cmdclass={"build_py": MyInstallData},
)
