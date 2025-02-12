import contextlib
import math

import cupy as cp
import mlir.extras.types as T
import numpy as np
from cupy.cuda import Module
from mlir.dialects import builtin

from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.context import (
    mlir_mod_ctx,
    MLIRContext,
)
from mlir.extras.dialects.ext import arith, memref, gpu, scf, linalg, vector, nvgpu
from mlir.extras.dialects.ext.gpu import (
    block_idx,
    thread_idx,
    block_dim,
    get_compile_object_bytes,
    smem_space,
)
from mlir.extras.dialects.ext.llvm import llvm_ptr_t
from mlir.extras.dialects.ext.memref import S
from mlir.extras.dialects.ext.scf import range_
from mlir.extras.runtime.passes import Pipeline, run_pipeline

# noinspection PyUnresolvedReferences
from mlir.extras.util import find_ops, enable_debug as enable_debug

# just so it doesn't get DCE'd by black/reformat
_ = memref


def build_cuda_func(compiled_module, kernel_name="naive"):
    ptx = get_compile_object_bytes(compiled_module)
    mod = Module()
    mod.load(ptx)
    return mod.get_function(kernel_name)


def print_ptx(compiled_module):
    ptx = get_compile_object_bytes(compiled_module)
    print(ptx.decode())


def compile_module(
    module,
    chip="sm_80",
    features="+ptx83",
    opt_level=2,
    enable_ir_printing=False,
    print_ptx_=False,
    full_pipeline=True,
):
    if enable_ir_printing:
        print_ptx_ = True
    if full_pipeline:
        p = (
            Pipeline()
            .convert_linalg_to_loops()
            .convert_nvgpu_to_nvvm()
            .gpu_kernel_outlining()
            .convert_vector_to_scf()
            .convert_scf_to_cf()
            .convert_nvvm_to_llvm()
            .convert_func_to_llvm()
            .expand_strided_metadata()
            .add_pass(
                "nvvm-attach-target",
                **{
                    "chip": chip,
                    "features": features,
                    "O": str(opt_level),
                },
            )
            .lower_affine()
            .convert_arith_to_llvm()
            .convert_index_to_llvm()
            .canonicalize()
            .cse()
            .Gpu(
                Pipeline()
                .strip_debuginfo()
                # TODO(max): upstream this (add to gpu pipeline)
                # vector.transfer
                .convert_vector_to_llvm()
                .convert_gpu_to_nvvm(use_bare_ptr_memref_call_conv=True)
                .canonicalize()
                .cse()
                .reconcile_unrealized_casts()
            )
            .gpu_to_llvm(use_bare_pointers_for_kernels=True)
            .gpu_module_to_binary(format="isa")
            .canonicalize()
            .cse()
            .reconcile_unrealized_casts()
        )
    else:
        p = Pipeline().add_pass(
            "gpu-lower-to-nvvm-pipeline",
            # https://github.com/llvm/llvm-project/blob/ace69e6b942b8fa7e610d70be2a92e801ceea481/mlir/include/mlir/Dialect/GPU/Pipelines/Passes.h#L18
            **{
                "cubin-chip": chip,
                "cubin-features": features,
                "cubin-format": "isa",
                "kernel-bare-ptr-calling-convention": "1",
                "opt-level": str(opt_level),
                # "cubin-format": "fatbin",
                # "cubin-format": "bin",
            },
        )
    mod = run_pipeline(module, p, enable_ir_printing=enable_ir_printing)

    if print_ptx_:
        print_ptx(mod)

    return mod


@contextlib.contextmanager
def time_cuda():
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()

    start_gpu.record()
    yield start_gpu, end_gpu
    end_gpu.record()
    end_gpu.synchronize()


@gpu.func
@canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
def sgemm_naive[
    M,
    K,
    N,
    dtype,
    A_t: T.memref(M, K, dtype),
    B_t: T.memref(K, N, dtype),
    C_t: T.memref(M, N, dtype),
](A: A_t, B: B_t, C: C_t):
    tmp = arith.constant(0, type=dtype)
    # this is from the example and it's basically a mistake
    # it increments the row for each adjacent thread id
    # uncomment the print to see
    r = block_dim.x * block_idx.x + thread_idx.x
    c = block_dim.y * block_idx.y + thread_idx.y
    # tid = gpu.thread_id()
    # gpu.printf("tid: %ld: (%ld, %ld)\n", tid, r, c)

    for k, tmp in range_(K, iter_args=[tmp]):
        tmp += A[r, k] * B[k, c]
        tmp = yield tmp
    C[r, c] = tmp + 1


@gpu.func
@canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
def sgemm_naive_row_order[
    M,
    K,
    N,
    dtype,
    A_t: T.memref(M, K, dtype),
    B_t: T.memref(K, N, dtype),
    C_t: T.memref(M, N, dtype),
](A: A_t, B: B_t, C: C_t):

    tmp = arith.constant(0, type=dtype)

    # increment along the cols (ie preserve row-order access)
    c = block_dim.x * block_idx.x + thread_idx.x
    r = block_dim.y * block_idx.y + thread_idx.y
    # tid = gpu.thread_id()
    # gpu.printf("tid: %ld: (%ld, %ld)\n", tid, r, c)

    for k, tmp in range_(K, iter_args=[tmp]):
        tmp += A[r, k] * B[k, c]
        tmp = yield tmp
    C[r, c] = tmp + 1


@gpu.func
@canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
def sgemm_coalesce[
    M,
    K,
    N,
    dtype,
    BLOCK_SIZE: 32,
    A_t: T.memref(M, K, dtype),
    B_t: T.memref(K, N, dtype),
    C_t: T.memref(M, N, dtype),
](A: A_t, B: B_t, C: C_t):

    tid = gpu.thread_id()
    # this is actually floordiv
    r = block_idx.x * BLOCK_SIZE + (tid / BLOCK_SIZE)
    c = block_idx.y * BLOCK_SIZE + (tid % BLOCK_SIZE)
    # gpu.printf("tid: %ld: (%ld, %ld)\n", tid, r, c)

    tmp = arith.constant(0, type=dtype)

    for k, tmp in range_(K, iter_args=[tmp]):
        # k varies per core while c varies with tid
        # apparently that's fine? i guess all the loads can happen
        # because there's enough scratch per SM to prefetch all the data each thread needs?
        tmp += A[r, k] * B[k, c]
        tmp = yield tmp
    C[r, c] = tmp + 1


# So if you try to load something like:
#
# B.T:
#
# 0 0 0 0 0 0 0 0
# 1 1 1 1 1 1 1 1
# 2 2 2 2 2 2 2 2
#
# vs
#
# B:
# 0 1 2 3 4 5 6 7 8
# 0 1 2 3 4 5 6 7 8
# 0 1 2 3 4 5 6 7 8
#
# In B, you are feeding all threads with a single load (say warp can load 8 elements at a time) and then you increment k
#
# in B.T, a single load is feeding only a single thread, so others are probably waiting for their load to happen
# these are the issues by threads:
#
# 0: (0, 0), (1, 0), (2, 0)
# 1: (0, 1), (1, 1), (2, 1)
# 2: (0, 2), (1, 2), (2, 2)
#
# warp recieves these issues:
#
# (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)
#
# warp issues coalesced reads:
#
# (0, 0:2), (1, 0:2), (2,0:2)
# so even though the threads have bad memory access pattern
# the warp has good memory access pattern
# and since the actual load happens at warp level
# its good
@gpu.func
@canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
def sgemm_coalesce_transpose_B[
    M,
    K,
    N,
    dtype,
    BLOCK_SIZE: 32,
    A_t: T.memref(M, K, dtype),
    B_t: T.memref(K, N, dtype),
    C_t: T.memref(M, N, dtype),
](A: A_t, B: B_t, C: C_t):

    tid = gpu.thread_id()
    r = block_idx.x * BLOCK_SIZE + (tid / BLOCK_SIZE)
    c = block_idx.y * BLOCK_SIZE + (tid % BLOCK_SIZE)

    tmp = arith.constant(0, type=dtype)

    for k, tmp in range_(K, iter_args=[tmp]):
        # this is slower because c is incremented with each tid
        # so you break memory coalescing
        # but k now being on the row order dim doesn't help?
        tmp += A[r, k] * B[c, k]
        tmp = yield tmp
    C[r, c] = tmp + 1


@gpu.func
@canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
def sgemm_shared_mem_block[
    M,
    K,
    N,
    dtype,
    BLOCK_SIZE: 32,
    A_t: T.memref(M, K, dtype),
    B_t: T.memref(K, N, dtype),
    C_t: T.memref(M, N, dtype),
](A: A_t, B: B_t, C: C_t):
    # allocate buffer for current block in fast shared mem
    # shared mem is shared between all threads in a block
    base = gpu.dynamic_shared_memory()
    A_shared = memref.view(base, (BLOCK_SIZE, BLOCK_SIZE), dtype=dtype)
    B_shared = memref.view(
        base, (BLOCK_SIZE, BLOCK_SIZE), dtype=dtype, shift=BLOCK_SIZE * BLOCK_SIZE
    )

    # the inner row & col that we're accessing in this thread
    tid = gpu.thread_id()
    thread_row = tid / BLOCK_SIZE
    thread_col = tid % BLOCK_SIZE

    # the output block that we want to compute in this threadblock
    c_row = block_idx.x * BLOCK_SIZE
    c_col = block_idx.y * BLOCK_SIZE

    tmp = arith.constant(0, type=dtype)

    for bk_idx, tmp in range_(0, K, BLOCK_SIZE, iter_args=[tmp]):
        A_ = A[c_row : c_row + BLOCK_SIZE, bk_idx : bk_idx + BLOCK_SIZE]
        B_ = B[bk_idx : bk_idx + BLOCK_SIZE, c_col : c_col + BLOCK_SIZE]

        # Have each thread load one of the elements in A & B
        # Make the threadCol (=thread_idx.x) the consecutive index
        # to allow global memory access coalescing
        A_shared[thread_row, thread_col] = A_[thread_row, thread_col]
        B_shared[thread_row, thread_col] = B_[thread_row, thread_col]

        # block threads in this block until cache is fully populated
        gpu.barrier()

        # execute the dotproduct on the currently cached block
        for dot_idx, tmp in range_(BLOCK_SIZE, iter_args=[tmp]):
            tmp += A_shared[thread_row, dot_idx] * B_shared[dot_idx, thread_col]
            tmp = yield tmp

        # need to sync again at the end, to avoid faster threads
        # fetching the next block into the cache before slower threads are done
        gpu.barrier()

        tmp = yield tmp

    C_ = C[c_row : c_row + BLOCK_SIZE, c_col : c_col + BLOCK_SIZE]
    C_[thread_row, thread_col] = tmp + 1


def prepare_non_tiled_kernel(ctx: MLIRContext, kernel, M, K, N, BLOCK_SIZE=32):
    dtype = T.f32()
    npy_dtype = np.float32

    gpu.set_container_module(ctx.module)

    @gpu.module("matmul", ["#nvvm.target"])
    def matmul_mod():
        kernel[M, K, N, dtype].emit()

    # print(ctx.module)
    # print(ctx.module.operation.verify())
    # exit()

    kernel_name = kernel.__name__
    compiled_module = compile_module(ctx.module)
    cuda_func = build_cuda_func(compiled_module, kernel_name)
    # print_ptx(compiled_module)

    grid_dims = (math.ceil(M / BLOCK_SIZE), math.ceil(N / BLOCK_SIZE))
    block_dims = (BLOCK_SIZE, BLOCK_SIZE)

    if "shared" in kernel_name:
        shared_mem = 2 * BLOCK_SIZE * BLOCK_SIZE * npy_dtype().nbytes
    else:
        shared_mem = 0

    return (
        cuda_func,
        grid_dims,
        block_dims,
        shared_mem,
        npy_dtype,
        "transpose_B" in kernel_name,
    )


@gpu.func
@canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
def sgemm_shared_mem_1d_block_tiling[
    M,
    K,
    N,
    dtype,
    BM,
    BN,
    BK,
    TM,
    A_t: T.memref(M, K, dtype),
    B_t: T.memref(K, N, dtype),
    C_t: T.memref(M, N, dtype),
](A: A_t, B: B_t, C: C_t):
    base = gpu.dynamic_shared_memory()
    A_shared = memref.view(base, (BM, BK), dtype=dtype)
    B_shared = memref.view(base, (BK, BN), dtype=dtype, shift=BM * BK)

    c_row = block_idx.y * BM
    c_col = block_idx.x * BN

    tid = gpu.thread_id()
    thread_col = tid % BN
    thread_row = tid / BN

    inner_col_A = tid % BK  # warp-level GMEM coalescing
    inner_row_A = tid / BK
    inner_col_B = tid % BN  # warp-level GMEM coalescing
    inner_row_B = tid / BN

    thread_results = memref.alloca((TM,), dtype)
    linalg.fill(0, thread_results)

    for bk_idx in range_(0, K, BK):
        # Move blocktile to beginning of A's row and B's column
        A_ = A[c_row : c_row + BM, bk_idx : bk_idx + BK]
        B_ = B[bk_idx : bk_idx + BK, c_col : c_col + BN]

        A_shared[inner_row_A, inner_col_A] = A_[inner_row_A, inner_col_A]
        B_shared[inner_row_B, inner_col_B] = B_[inner_row_B, inner_col_B]

        gpu.barrier()

        for dot_idx in range_(BK):
            tmp_B = B_shared[dot_idx, thread_col]
            for res_idx, tmp_B in range_(TM, iter_args=[tmp_B]):
                thread_results[res_idx] += (
                    A_shared[thread_row * TM + res_idx, dot_idx] * tmp_B
                )
                yield tmp_B

        gpu.barrier()

    C_ = C[c_row : c_row + BM, c_col : c_col + BN]
    for res_idx in range_(TM):
        C_[thread_row * TM + res_idx, thread_col] = thread_results[res_idx] + 1


@gpu.func
@canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
def sgemm_shared_mem_2d_block_tiling[
    M,
    K,
    N,
    dtype,
    BM,
    BN,
    BK,
    TM,
    TN,
    A_t: T.memref(M, K, dtype),
    B_t: T.memref(K, N, dtype),
    C_t: T.memref(M, N, dtype),
](A: A_t, B: B_t, C: C_t):
    base = gpu.dynamic_shared_memory()
    A_shared = memref.view(base, (BM, BK), dtype=dtype)
    B_shared = memref.view(base, (BK, BN), dtype=dtype, shift=BM * BK)

    c_row = block_idx.y * BM
    c_col = block_idx.x * BN

    total_results_blocktile = BM * BN
    num_threads_blocktile = total_results_blocktile // (TM * TN)

    tid = gpu.thread_id()
    # BN/TN are the number of threads to span a column
    thread_col = tid % (BN // TN)
    thread_row = tid / (BN // TN)

    inner_col_A = tid % BK  # warp-level GMEM coalescing
    inner_row_A = tid / BK
    stride_A = num_threads_blocktile // BK

    inner_col_B = tid % BN  # warp-level GMEM coalescing
    inner_row_B = tid / BN
    stride_B = num_threads_blocktile // BN

    thread_results = memref.alloca((TM, TN), dtype)
    linalg.fill(0, thread_results)

    reg_M = memref.alloca((TM,), dtype)
    linalg.fill(0, reg_M)

    reg_N = memref.alloca((TN,), dtype)
    linalg.fill(0, reg_N)

    for bk_idx in range_(0, K, BK):
        A_ = A[c_row : c_row + BM, bk_idx : bk_idx + BK]
        B_ = B[bk_idx : bk_idx + BK, c_col : c_col + BN]

        for load_offset in range_(0, BM, stride_A):
            A_shared[inner_row_A + load_offset, inner_col_A] = A_[
                inner_row_A + load_offset, inner_col_A
            ]
        for load_offset in range_(0, BK, stride_B):
            B_shared[inner_row_B + load_offset, inner_col_B] = B_[
                inner_row_B + load_offset, inner_col_B
            ]

        gpu.barrier()

        for dot_idx in range_(BK):
            for i in range_(TM):
                reg_M[i] = A_shared[thread_row * TM + i, dot_idx]
            for i in range_(TN):
                reg_N[i] = B_shared[dot_idx, thread_col * TN + i]

            for res_idx_m in range_(TM):
                for res_idx_n in range_(TN):
                    thread_results[res_idx_m, res_idx_n] += (
                        reg_M[res_idx_m] * reg_N[res_idx_n]
                    )

        gpu.barrier()

    C_ = C[c_row : c_row + BM, c_col : c_col + BN]

    for res_idx_m in range_(TM):
        for res_idx_n in range_(TN):
            C_[thread_row * TM + res_idx_m, thread_col * TN + res_idx_n] = (
                thread_results[res_idx_m, res_idx_n] + 1
            )


@gpu.func
@canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
def sgemm_shared_mem_2d_block_tiling_vectorize[
    M,
    K,
    N,
    dtype,
    BM,
    BN,
    BK,
    TM,
    TN,
    A_t: T.memref(M, K, dtype),
    B_t: T.memref(K, N, dtype),
    C_t: T.memref(M, N, dtype),
](A: A_t, B: B_t, C: C_t):
    VECTOR_WIDTH = 4
    DTYPE_WIDTH = dtype.width // 8

    # ld.global.v4.u32 and st.global.v4.f32 emitted only input args are aligned
    # alignment for cupy is 512 bytes https://github.com/cupy/cupy/blob/59e6c2b2e0c722b09c7a7af13f908942ef7806cc/cupy/cuda/memory.pyx#L805-L809
    # so we're good
    memref.assume_alignment(A, VECTOR_WIDTH * DTYPE_WIDTH)
    memref.assume_alignment(B, VECTOR_WIDTH * DTYPE_WIDTH)
    memref.assume_alignment(C, VECTOR_WIDTH * DTYPE_WIDTH)

    base = gpu.dynamic_shared_memory()
    base = memref.memory_space_cast(T.memref(S, element_type=T.i8()), base)

    # transpose A
    A_shared = memref.view(base, (BK, BM), dtype=dtype)
    B_shared = memref.view(base, (BK, BN), dtype=dtype, shift=BM * BK)

    c_row = block_idx.y * BM
    c_col = block_idx.x * BN

    tid = gpu.thread_id()
    # BN/TN are the number of threads to span a column
    thread_col = tid % (BN // TN)
    thread_row = tid / (BN // TN)

    # calculating the indices that this thread will load into SMEM
    # we'll load 128bit / 32bit = 4 elements per thread at each step
    inner_col_A = tid % (BK // VECTOR_WIDTH)  # warp-level GMEM coalescing
    inner_row_A = tid / (BK // VECTOR_WIDTH)
    inner_col_B = tid % (BN // VECTOR_WIDTH)  # warp-level GMEM coalescing
    inner_row_B = tid / (BN // VECTOR_WIDTH)

    thread_results = memref.alloca((TM, TN), dtype)
    linalg.fill(0, thread_results)

    reg_M = memref.alloca((TM,), dtype)
    linalg.fill(0, reg_M)

    reg_N = memref.alloca((TN,), dtype)
    linalg.fill(0, reg_N)

    for bk_idx in range_(0, K, BK):
        A_ = A[c_row : c_row + BM, bk_idx : bk_idx + BK]
        B_ = B[bk_idx : bk_idx + BK, c_col : c_col + BN]

        A_vec = vector.load(
            T.vector(VECTOR_WIDTH, dtype), A_, [inner_row_A, inner_col_A * VECTOR_WIDTH]
        )
        for j in range(VECTOR_WIDTH):
            #  transpose A while loading it
            A_shared[inner_col_A * VECTOR_WIDTH + j, inner_row_A] = A_vec[j]

        B_vec = vector.load(
            T.vector(VECTOR_WIDTH, dtype), B_, [inner_row_B, inner_col_B * VECTOR_WIDTH]
        )
        vector.store(B_vec, B_shared, [inner_row_B, inner_col_B * VECTOR_WIDTH])

        gpu.barrier()

        for dot_idx in range_(BK):
            for i in range_(TM):
                reg_M[i] = A_shared[dot_idx, thread_row * TM + i]

            for i in range_(TN):
                reg_N[i] = B_shared[dot_idx, thread_col * TN + i]

            for res_idx_m in range_(TM):
                for res_idx_n in range_(TN):
                    thread_results[res_idx_m, res_idx_n] += (
                        reg_M[res_idx_m] * reg_N[res_idx_n]
                    )

        gpu.barrier()

    C_ = C[c_row : c_row + BM, c_col : c_col + BN]

    for res_idx_m in range_(TM):
        for res_idx_n in range_(0, TN, VECTOR_WIDTH):
            tmp = vector.load(
                T.vector(VECTOR_WIDTH, dtype),
                C_,
                [thread_row * TM + res_idx_m, thread_col * TN + res_idx_n],
            )
            for j in range(VECTOR_WIDTH):
                tmp[j] = thread_results[res_idx_m, res_idx_n + j] + 1
            vector.store(
                tmp, C_, [thread_row * TM + res_idx_m, thread_col * TN + res_idx_n]
            )


WARP_SIZE = 32


@gpu.func
@canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
def sgemm_warp_tiling[
    M,
    K,
    N,
    dtype,
    BM,
    BN,
    BK,
    WM,
    WN,
    WNITER,
    TM,
    TN,
    NUM_THREADS,
    A_t: T.memref(M, K, dtype),
    B_t: T.memref(K, N, dtype),
    C_t: T.memref(M, N, dtype),
](A: A_t, B: B_t, C: C_t):
    VECTOR_WIDTH = 4
    DTYPE_WIDTH = dtype.width // 8

    tid = gpu.thread_id()

    # ld.global.v4.u32 and st.global.v4.f32 emitted only input args are aligned
    # alignment for cupy is 512 bytes https://github.com/cupy/cupy/blob/59e6c2b2e0c722b09c7a7af13f908942ef7806cc/cupy/cuda/memory.pyx#L805-L809
    # so we're good
    memref.assume_alignment(A, VECTOR_WIDTH * DTYPE_WIDTH)
    memref.assume_alignment(B, VECTOR_WIDTH * DTYPE_WIDTH)
    memref.assume_alignment(C, VECTOR_WIDTH * DTYPE_WIDTH)

    base = gpu.dynamic_shared_memory()
    base = memref.memory_space_cast(T.memref(S, element_type=T.i8()), base)

    # transpose A
    A_shared = memref.view(base, (BK, BM), dtype=dtype)
    B_shared = memref.view(base, (BK, BN), dtype=dtype, shift=BM * BK)

    c_row = block_idx.y * BM
    c_col = block_idx.x * BN

    # Placement of the warp in the threadblock tile
    warp_idx = tid / WARP_SIZE
    warp_row = warp_idx / (BN // WN)
    warp_col = warp_idx % (BN // WN)

    # size of the warp subtile
    WMITER = (WM * WN) // (WARP_SIZE * TM * TN * WNITER)
    WSUBM = WM // WMITER
    WSUBN = WN // WNITER

    # Placement of the thread in the warp subtile
    thread_idx_in_warp = tid % WARP_SIZE
    thread_col_in_warp = thread_idx_in_warp % (WSUBN // TN)
    thread_row_in_warp = thread_idx_in_warp / (WSUBN // TN)

    # calculating the indices that this thread will load into SMEM
    # we'll load 128bit / 32bit = 4 elements per thread at each step
    inner_row_A = tid / (BK // VECTOR_WIDTH)
    inner_col_A = tid % (BK // VECTOR_WIDTH)
    row_stride_A = (NUM_THREADS * VECTOR_WIDTH) // BK
    inner_row_B = tid / (BN // VECTOR_WIDTH)
    inner_col_B = tid % (BN // VECTOR_WIDTH)
    row_stride_B = NUM_THREADS // (BN // VECTOR_WIDTH)

    # allocate thread-local cache for results in registerfile
    thread_results = memref.alloca((WMITER * TM, WNITER * TN), dtype)
    linalg.fill(0, thread_results)

    reg_M = memref.alloca((WMITER, TM), dtype)
    linalg.fill(0, reg_M)

    reg_N = memref.alloca((WNITER, TN), dtype)
    linalg.fill(0, reg_N)

    for bk_idx in range_(0, K, BK):
        A_ = A[c_row : c_row + BM, bk_idx : bk_idx + BK]
        B_ = B[bk_idx : bk_idx + BK, c_col : c_col + BN]

        for offset in range(0, BM - row_stride_A + 1, row_stride_A):
            A_vec = vector.load(
                T.vector(VECTOR_WIDTH, dtype),
                A_,
                [inner_row_A + offset, inner_col_A * VECTOR_WIDTH],
            )
            for j in range(VECTOR_WIDTH):
                #  transpose A while loading it
                A_shared[inner_col_A * VECTOR_WIDTH + j, inner_row_A + offset] = A_vec[
                    j
                ]

        for offset in range(0, BK - row_stride_B + 1, row_stride_B):
            B_vec = vector.load(
                T.vector(VECTOR_WIDTH, dtype),
                B_,
                [inner_row_B + offset, inner_col_B * VECTOR_WIDTH],
            )
            vector.store(
                B_vec, B_shared, [inner_row_B + offset, inner_col_B * VECTOR_WIDTH]
            )

        gpu.barrier()

        for dot_idx in range_(BK):
            for w_sub_row_idx in range_(WMITER):
                for i in range_(TM):
                    reg_M[w_sub_row_idx, i] = A_shared[
                        dot_idx,
                        warp_row * WM
                        + w_sub_row_idx * WSUBM
                        + thread_row_in_warp * TM
                        + i,
                    ]

            for w_sub_col_idx in range_(WNITER):
                for i in range_(TN):
                    reg_N[w_sub_col_idx, i] = B_shared[
                        dot_idx,
                        warp_col * WN
                        + w_sub_col_idx * WSUBN
                        + thread_col_in_warp * TN
                        + i,
                    ]

            for w_sub_row_idx in range_(WMITER):
                for w_sub_col_idx in range_(WNITER):
                    for res_idx_m in range_(TM):
                        for res_idx_n in range_(TN):
                            thread_results[
                                w_sub_row_idx * TM + res_idx_m,
                                w_sub_col_idx * TN + res_idx_n,
                            ] += (
                                reg_M[w_sub_row_idx, res_idx_m]
                                * reg_N[w_sub_col_idx, res_idx_n]
                            )

        gpu.barrier()

    for w_sub_row_idx in range_(WMITER):
        for w_sub_col_idx in range_(WNITER):
            r = c_row + warp_row * WM + w_sub_row_idx * WSUBM
            c = c_col + warp_col * WN + w_sub_col_idx * WSUBN
            C_ = C[r : r + WSUBM, c : c + WSUBN]
            for res_idx_m in range_(TM):
                for res_idx_n in range_(0, TN, VECTOR_WIDTH):
                    tmp = vector.load(
                        T.vector(VECTOR_WIDTH, dtype),
                        C_,
                        [
                            thread_row_in_warp * TM + res_idx_m,
                            thread_col_in_warp * TN + res_idx_n,
                        ],
                    )
                    for j in range(VECTOR_WIDTH):
                        tmp[j] = (
                            thread_results[
                                w_sub_row_idx * TM + res_idx_m,
                                w_sub_col_idx * TN + res_idx_n + j,
                            ]
                            + 1
                        )
                    vector.store(
                        tmp,
                        C_,
                        [
                            thread_row_in_warp * TM + res_idx_m,
                            thread_col_in_warp * TN + res_idx_n,
                        ],
                    )


@gpu.func
@canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
def sgemm_tensor_core[
    M,
    K,
    N,
    A_t: T.memref(M, K, T.f16()),
    B_t: T.memref(K, N, T.f16()),
    C_t: T.memref(M, N, T.f32()),
    a_tma_t: llvm_ptr_t(),
    b_tma_t: llvm_ptr_t(),
](A: A_t, B: B_t, C: C_t, a_tma: a_tma_t, b_tma: b_tma_t):
    a_tma = builtin.unrealized_conversion_cast(
        [
            nvgpu.TensorMapDescriptorType.get(
                T.memref(128, 64, T.f16(), memory_space=smem_space()),
                swizzle=int(nvgpu.TensorMapSwizzleKind.SWIZZLE_128B),
                l2promo=int(nvgpu.TensorMapL2PromoKind.L2PROMO_NONE),
                oob_fill=int(nvgpu.TensorMapOOBKind.OOB_ZERO),
                interleave=int(nvgpu.TensorMapInterleaveKind.INTERLEAVE_NONE),
            )
        ],
        [a_tma],
    )
    b_tma = builtin.unrealized_conversion_cast(
        [
            nvgpu.TensorMapDescriptorType.get(
                T.memref(64, 64, T.f16(), memory_space=smem_space()),
                swizzle=int(nvgpu.TensorMapSwizzleKind.SWIZZLE_128B),
                l2promo=int(nvgpu.TensorMapL2PromoKind.L2PROMO_NONE),
                oob_fill=int(nvgpu.TensorMapOOBKind.OOB_ZERO),
                interleave=int(nvgpu.TensorMapInterleaveKind.INTERLEAVE_NONE),
            )
        ],
        [b_tma],
    )
    tid = gpu.thread_id()
    is_thread_0 = tid == 0

    mbarrier = nvgpu.mbarrier_create()
    nvgpu.mbarrier_init(mbarrier, 1, 0, predicate=is_thread_0)
    nvgpu.tma_prefetch_descriptor(a_tma)
    nvgpu.tma_prefetch_descriptor(b_tma)

    base = gpu.dynamic_shared_memory()

    shift = 0
    A_shared = memref.view(base, (M, K), dtype=T.f16(), shift=shift)
    shift += A_shared.n_elements
    B_shared = memref.view(base, (K, N), dtype=T.f16(), shift=shift)
    shift += B_shared.n_elements

    a = memref.view(base, (128, 64), dtype=T.f16(), shift=shift)
    shift += a.n_elements
    b1 = memref.view(base, (64, 64), dtype=T.f16(), shift=shift)
    shift += b1.n_elements
    b2 = memref.view(base, (64, 64), dtype=T.f16(), shift=shift)

    ta_count = a.n_elements + b1.n_elements + b2.n_elements
    nvgpu.mbarrier_arrive_expect_tx(mbarrier, ta_count, 0, predicate=is_thread_0)

    nvgpu.tma_async_load(
        a,
        mbarrier,
        a_tma,
        coordinates=[0, 0],
        mbar_id=0,
        predicate=is_thread_0,
    )
    nvgpu.tma_async_load(
        b1,
        mbarrier,
        b_tma,
        coordinates=[0, 0],
        mbar_id=0,
        predicate=is_thread_0,
    )
    nvgpu.tma_async_load(
        b2,
        mbarrier,
        b_tma,
        coordinates=[64, 0],
        mbar_id=0,
        predicate=is_thread_0,
    )
    nvgpu.mbarrier_try_wait_parity(mbarrier, mbar_id=0)

    accum = nvgpu.warpgroup_mma_init_accumulator(
        nvgpu.warpgroup_accumulator_t(M, N, T.f32())
    )
    lhs = nvgpu.warpgroup_generate_descriptor(
        nvgpu.warpgroup_descriptor(M, K, T.f16()), A_shared, a_tma
    )
    rhs = nvgpu.warpgroup_generate_descriptor(
        nvgpu.warpgroup_descriptor(K, N, T.f16()), B_shared, b_tma
    )
    acc = nvgpu.warpgroup_mma(accum, lhs, rhs, transpose_b=True)
    nvgpu.warpgroup_mma_store(acc, C)


def prepare_tiled_kernel(ctx: MLIRContext, kernel, M, K, N):
    dtype = T.f32()
    npy_dtype = np.float32
    kernel_name = kernel.__name__

    gpu.set_container_module(ctx.module)

    BK = 8
    TM = 8
    TN = 8
    if "2d" in kernel_name and M >= 128 and N >= 128:
        BM = 128
        BN = 128
    else:
        BM = 64
        BN = 64

    @gpu.module("matmul", ["#nvvm.target"])
    def matmul_mod():
        kernel[M, K, N, dtype, BM, BN, BK, TM, TN].emit()

    # print(ctx.module)
    # print(ctx.module.operation.verify())
    # exit()

    compiled_module = compile_module(ctx.module)
    cuda_func = build_cuda_func(compiled_module, kernel_name)
    # print_ptx(compiled_module)

    grid_dims = (math.ceil(N / BN), math.ceil(M / BM))
    if "2d" in kernel_name:
        block_dims = (BM // TM, BN // TN)
    else:
        block_dims = (BM // TM, BN)

    if "shared" in kernel_name:
        shared_mem = ((BM * BK) + (BK * BN)) * npy_dtype().nbytes
    else:
        shared_mem = 0

    return (
        cuda_func,
        grid_dims,
        block_dims,
        shared_mem,
        npy_dtype,
        False,
    )


def prepare_warp_tiled_kernel(ctx: MLIRContext, kernel, M, K, N):
    dtype = T.f32()
    npy_dtype = np.float32
    kernel_name = kernel.__name__

    gpu.set_container_module(ctx.module)

    # Settings for A100 (looks like it works for 3070 too?)
    NUM_THREADS = 128
    BN = 128
    BM = 64
    BK = 16
    WN = 64
    WM = 32
    WNITER = 1
    TN = 4
    TM = 4

    @gpu.module("matmul", ["#nvvm.target"])
    def matmul_mod():
        kernel[M, K, N, dtype, BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS].emit()

    # print(ctx.module)
    # print(ctx.module.operation.verify())
    # exit()

    compiled_module = compile_module(ctx.module)
    cuda_func = build_cuda_func(compiled_module, kernel_name)
    # print_ptx(compiled_module)

    grid_dims = (math.ceil(N / BN), math.ceil(M / BM))
    block_dims = (NUM_THREADS,)
    shared_mem = ((BM * BK) + (BK * BN)) * npy_dtype().nbytes

    return (
        cuda_func,
        grid_dims,
        block_dims,
        shared_mem,
        npy_dtype,
        False,
    )


def prepare_tensor_core_kernel(ctx: MLIRContext, kernel, M, K, N):
    dtype = T.f16()
    npy_dtype = np.float16
    kernel_name = kernel.__name__

    gpu.set_container_module(ctx.module)

    # Settings for A100 (looks like it works for 3070 too?)
    NUM_THREADS = 128
    BN = 128
    BM = 64
    BK = 16
    WN = 64
    WM = 32
    WNITER = 1
    TN = 4
    TM = 4

    @gpu.module("matmul", ["#nvvm.target"])
    def matmul_mod():
        kernel[M, K, N, dtype].emit()

    print(ctx.module)
    print(ctx.module.operation.verify())
    # exit()

    compiled_module = compile_module(
        ctx.module, chip="sm_90a", opt_level=3, full_pipeline=False
    )
    # cuda_func = build_cuda_func(compiled_module, kernel_name)
    # print_ptx(compiled_module)

    grid_dims = (math.ceil(N / BN), math.ceil(M / BM))
    block_dims = (NUM_THREADS,)
    shared_mem = ((BM * BK) + (BK * BN)) * npy_dtype().nbytes

    return (
        # cuda_func,
        None,
        grid_dims,
        block_dims,
        shared_mem,
        npy_dtype,
        False,
    )


def run_eval(
    M,
    K,
    N,
    cuda_func,
    grid_dims,
    block_dims,
    shared_mem,
    npy_dtype,
    transpose_B,
    repeat_times=None,
):
    if repeat_times is None:
        repeat_times = 50

    A = np.random.randint(0, 10, (M, K)).astype(npy_dtype)
    B = np.random.randint(0, 10, (K, N)).astype(npy_dtype)
    C = np.zeros((M, N)).astype(npy_dtype)

    dA = cp.asarray(A)
    if transpose_B:
        dB = cp.asarray(np.ascontiguousarray(B.T))
    else:
        dB = cp.asarray(B)
    dC = cp.asarray(C)

    cuda_func(
        grid_dims,
        block_dims,
        (dA.data.ptr, dB.data.ptr, dC.data.ptr),
        shared_mem=shared_mem,
    )
    C = cp.asnumpy(dC)
    if not np.array_equal(C, A @ B + 1):
        print(A @ B + 1)
        print(C)
        assert False
    if repeat_times < 1:
        return

    with time_cuda() as (start_gpu, end_gpu):
        for _ in range(repeat_times):
            cuda_func(
                grid_dims,
                block_dims,
                (dA.data.ptr, dB.data.ptr, dC.data.ptr),
                shared_mem=shared_mem,
            )

    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)

    print(f"t={t_gpu / repeat_times:.6f} ms")


sizes = [128, 256, 512, 1024]
repeats = None

for k in [
    sgemm_naive,
    sgemm_naive_row_order,
    sgemm_coalesce,
    sgemm_coalesce_transpose_B,
    sgemm_shared_mem_block,
]:
    print(f"\n{k.__name__}")
    for s in sizes:
        with (
            mlir_mod_ctx() as ctx,
            # enable_debug()
        ):
            print(f"{s=}", end=" ")
            cuda_func, grid_dims, block_dims, shared_mem, npy_dtype, transpose_B = (
                prepare_non_tiled_kernel(ctx, k, s, s, s)
            )
            run_eval(
                s,
                s,
                s,
                cuda_func,
                grid_dims,
                block_dims,
                shared_mem,
                npy_dtype,
                transpose_B,
            )


for k in [
    sgemm_shared_mem_1d_block_tiling,
    sgemm_shared_mem_2d_block_tiling,
    sgemm_shared_mem_2d_block_tiling_vectorize,
]:
    print(f"\n{k.__name__}")
    for s in sizes:
        with (
            mlir_mod_ctx() as ctx,
            # enable_debug()
        ):
            print(f"{s=}", end=" ")
            cuda_func, grid_dims, block_dims, shared_mem, npy_dtype, transpose_B = (
                prepare_tiled_kernel(ctx, k, s, s, s)
            )
            run_eval(
                s,
                s,
                s,
                cuda_func,
                grid_dims,
                block_dims,
                shared_mem,
                npy_dtype,
                transpose_B,
            )

print(f"\n{sgemm_warp_tiling.__name__}")
for s in sizes:
    with (
        mlir_mod_ctx() as ctx,
        # enable_debug()
    ):
        print(f"{s=}", end=" ")
        cuda_func, grid_dims, block_dims, shared_mem, npy_dtype, transpose_B = (
            prepare_warp_tiled_kernel(ctx, sgemm_warp_tiling, s, s, s)
        )
        run_eval(
            s,
            s,
            s,
            cuda_func,
            grid_dims,
            block_dims,
            shared_mem,
            npy_dtype,
            transpose_B,
        )


sizes = [128, 256]

for s in sizes:
    with (
        mlir_mod_ctx() as ctx,
        # enable_debug()
    ):
        print(f"{s=}", end=" ")
        cuda_func, grid_dims, block_dims, shared_mem, npy_dtype, transpose_B = (
            prepare_tensor_core_kernel(ctx, sgemm_tensor_core, s, s, s)
        )
        # run_eval(
        #     s,
        #     s,
        #     s,
        #     cuda_func,
        #     grid_dims,
        #     block_dims,
        #     shared_mem,
        #     npy_dtype,
        #     transpose_B,
        # )
