import ctypes
from functools import partial
from typing import Union

import numpy as np
from mlir.ir import (
    Attribute,
    BF16Type,
    ComplexType,
    F16Type,
    F32Type,
    F64Type,
    Float8E5M2Type,
    Float8E4M3FNType,
    Float8E4M3B11FNUZType,
    IndexType,
    IntegerType,
    MemRefType,
    NoneType,
    OpaqueType,
    RankedTensorType,
    Type,
    UnrankedMemRefType,
    UnrankedTensorType,
    VectorType,
)

index_t = IndexType.get()
bool_t = IntegerType.get_signless(1)

i8_t = IntegerType.get_signless(8)
i16_t = IntegerType.get_signless(16)
i32_t = IntegerType.get_signless(32)
i64_t = IntegerType.get_signless(64)

si8_t = IntegerType.get_signed(8)
si16_t = IntegerType.get_signed(16)
si32_t = IntegerType.get_signed(32)
si64_t = IntegerType.get_signed(64)

ui8_t = IntegerType.get_unsigned(8)
ui16_t = IntegerType.get_unsigned(16)
ui32_t = IntegerType.get_unsigned(32)
ui64_t = IntegerType.get_unsigned(64)

f16_t = F16Type.get()
f32_t = F32Type.get()
f64_t = F64Type.get()
bf16_t = BF16Type.get()
f8e5m2_t = Float8E5M2Type.get()
f8e4m3_t = Float8E4M3FNType.get()
f8e4m3b11fnuz_t = Float8E4M3B11FNUZType.get()

cmp16_t = ComplexType.get(f16_t)
cmp32_t = ComplexType.get(f32_t)
cmp64_t = ComplexType.get(f64_t)

opaque_t = lambda dialect_namespace, buffer: OpaqueType.get(dialect_namespace, buffer)
none_t = NoneType.get()

NP_DTYPE_TO_MLIR_TYPE = {
    np.int8: i8_t,
    np.int16: i16_t,
    np.int32: i32_t,
    # windows
    np.intc: i32_t,
    np.int64: i64_t,
    # this is techincally wrong i guess but numpy by default casts python scalars to this
    # so to support passing lists of ints we map this to index type
    np.longlong: index_t,
    np.uintp: index_t,
    np.float16: f16_t,
    np.float32: f32_t,
    np.float64: f64_t,
}

MLIR_TYPE_TO_NP_DTYPE = lambda: {v: k for k, v in NP_DTYPE_TO_MLIR_TYPE.items()}

MLIR_TYPE_TO_CTYPE = {
    bool_t: ctypes.c_bool,
    i8_t: ctypes.c_byte,
    i64_t: ctypes.c_int,
    f32_t: ctypes.c_float,
    f64_t: ctypes.c_double,
}

MLIR_TYPE_TO_CTYPE = {
    bool_t: ctypes.c_bool,
    i8_t: ctypes.c_byte,
    i64_t: ctypes.c_int,
    f32_t: ctypes.c_float,
    f64_t: ctypes.c_double,
}


def infer_mlir_type(
    py_val: Union[int, float, bool, np.ndarray]
) -> Union[IntegerType, F32Type, F64Type, RankedTensorType]:
    """Infer MLIR type (`ir.Type`) from supported python values.

    Note ints and floats are mapped to 64-bit types.

    Args:
      py_val: Python value that's either a numerical value or numpy array.

    Returns:
      MLIR type corresponding to py_val.
    """
    if isinstance(py_val, bool):
        return bool_t
    elif isinstance(py_val, int):
        if -(2 ** 31) <= py_val < 2 ** 31:
            return i32_t
        elif 2 ** 31 <= py_val < 2 ** 32:
            return ui32_t
        elif -(2 ** 63) <= py_val < 2 ** 63:
            return i64_t
        elif 2 ** 63 <= py_val < 2 ** 64:
            return ui64_t
        else:
            raise RuntimeError(f"Nonrepresentable integer {py_val}.")
    elif isinstance(py_val, float):
        if (
            abs(py_val) == float("inf")
            or abs(py_val) == 0.0
            or py_val != py_val  # NaN
            or np.finfo(np.float32).min <= abs(py_val) <= np.finfo(np.float32).max
        ):
            return f32_t
        else:
            return f64_t
    elif isinstance(py_val, np.ndarray):
        dtype = NP_DTYPE_TO_MLIR_TYPE[py_val.dtype.type]
        return RankedTensorType.get(py_val.shape, dtype)
    else:
        raise NotImplementedError(
            f"Unsupported Python value {py_val=} with type {type(py_val)}"
        )


def shaped_t(*args, element_type: Type = None, type_constructor=None):
    if type_constructor is None:
        raise ValueError("shaped_t is an abstract base class - cannot be constructed")
    if (element_type is None and args and not isinstance(args[-1], Type)) or (
        args and isinstance(args[-1], Type) and element_type is not None
    ):
        raise ValueError(
            f"either element_type must be provided explicitly XOR last arg to tensor type constructor must be the element type"
        )
    if element_type is not None:
        type = element_type
        sizes = args
    else:
        type = args[-1]
        sizes = args[:-1]
    if sizes:
        return type_constructor(sizes, type)
    else:
        return type_constructor(type)


def vector_t(*args, element_type: Type = None):
    return shaped_t(*args, element_type=element_type, type_constructor=VectorType.get)


def tensor_t(*args, element_type: Type = None):
    if not len(args) or len(args) == 1 and isinstance(args[-1], Type):
        return shaped_t(
            *args, element_type=element_type, type_constructor=UnrankedTensorType.get
        )
    else:
        return shaped_t(
            *args, element_type=element_type, type_constructor=RankedTensorType.get
        )


def memref_t(*args, element_type: Type = None, memory_space: int = None):
    if memory_space is None:
        memory_space = 0
    memory_space = Attribute.parse(str(memory_space))
    if not len(args) or len(args) == 1 and isinstance(args[-1], Type):
        return shaped_t(
            *args,
            element_type=element_type,
            type_constructor=partial(UnrankedMemRefType.get, memory_space=memory_space),
        )
    else:
        return shaped_t(
            *args,
            element_type=element_type,
            type_constructor=partial(MemRefType.get, memory_space=memory_space),
        )


MEMREF_TYPE_TO_NP_DTYPE = {
    memref_t(element_type=f16_t): np.float16,
    memref_t(element_type=f32_t): np.float32,
    memref_t(f64_t): np.float64,
    memref_t(element_type=bool_t): np.bool_,
    memref_t(i8_t): np.int8,
    memref_t(i32_t): np.int32,
    memref_t(i64_t): np.int64,
}
