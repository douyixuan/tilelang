# Compiling Kernels Without GPU Hardware

This feature (added for issue #1049) allows you to compile TileLang kernels to source code or binary without requiring GPU hardware to be present. This is useful for:

- **Code generation and inspection**: See how TileLang generates CUDA/HIP code
- **Cross-compilation**: Compile kernels on a system without GPU for deployment elsewhere
- **Frontend issue detection**: Catch issues in kernel code before running on hardware
- **CI/CD pipelines**: Build and test kernel compilation without GPU access

## Usage

### With the `@tilelang.jit` decorator

```python
import tilelang
import tilelang.language as T

@tilelang.jit(
    out_idx=[-1], 
    target="cuda",  # Specify target explicitly
    skip_hardware_check=True  # Skip GPU detection
)
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    @T.prim_func
    def gemm(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm

# Compile without GPU hardware
kernel = matmul(1024, 1024, 1024, 128, 128, 32)

# Get generated source code
cuda_source = kernel.get_kernel_source()
print(cuda_source)
```

### With the `tilelang.compile` function

```python
import tilelang
import tilelang.language as T

@T.prim_func
def my_kernel(...):
    ...

# Compile without GPU hardware
kernel = tilelang.compile(
    my_kernel,
    out_idx=[-1],
    target="cuda",  # or "hip" for AMD GPUs
    skip_hardware_check=True
)

# Get generated source code
source = kernel.get_kernel_source()
```

## Important Notes

1. **Target must be explicitly specified**: When using `skip_hardware_check=True`, you should explicitly specify the target (e.g., `target="cuda"` or `target="hip"`). If you use `target="auto"` with `skip_hardware_check=True`, it will default to CUDA.

2. **Source code only**: With `skip_hardware_check=True`, you can generate and inspect source code, but you cannot execute the kernel (since no GPU is present).

3. **Supported targets**: Currently supports:
   - `"cuda"` - NVIDIA CUDA
   - `"hip"` - AMD ROCm/HIP
   
4. **No execution**: Attempting to call the kernel (e.g., `kernel(a, b)`) will fail if no GPU hardware is available.

## Examples

See `test_no_gpu_compile.py` for a complete working example.

## Background

This feature was requested in [issue #1049](https://github.com/tile-ai/tilelang/issues/1049) to help kernel developers:
- See how code generation works without needing GPU access
- Detect potential frontend issues early in development
- Enable CI/CD workflows that don't have GPU access
