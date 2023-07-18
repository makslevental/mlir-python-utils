import os
from pathlib import Path
from textwrap import dedent

import pytest

import mlir_utils.dialects
from mlir_utils._configuration.generate_trampolines import (
    generate_trampolines,
)

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_smoke(ctx: MLIRContext):
    correct = dedent(
        """\
    module {
    }
    """
    )
    filecheck(correct, ctx.module)


def test_dialect_trampolines_smoke():
    # noinspection PyUnresolvedReferences
    from mlir_utils.dialects import (
        arith,
        bufferization,
        builtin,
        cf,
        complex,
        func,
        gpu,
        linalg,
        math,
        memref,
        ml_program,
        pdl,
        scf,
        shape,
        sparse_tensor,
        tensor,
        tosa,
        transform,
        vector,
    )


def skip_torch_mlir_not_installed():
    try:
        from torch_mlir.dialects import torch

        # don't skip
        return False
    except ImportError:
        # skip
        return True


@pytest.mark.skipif(skip_torch_mlir_not_installed(), reason="torch_mlir not installed")
def test_torch_dialect_trampolines_smoke():
    try:
        modu = __import__("mlir_utils.dialects.torch", fromlist=["*"])
        os.remove(modu.__file__)
    except ModuleNotFoundError:
        pass
    generate_trampolines(
        "torch_mlir.dialects.torch", Path(mlir_utils.dialects.__path__[0]), "torch"
    )
    # noinspection PyUnresolvedReferences
    from mlir_utils.dialects import torch
