from ....ir import Type


def llvm_ptr_t():
    return Type.parse("!llvm.ptr")
