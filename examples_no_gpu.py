#!/usr/bin/env python3
"""
Additional examples for compiling kernels without GPU hardware.

This demonstrates various use cases for the skip_hardware_check feature.
"""

import tilelang
import tilelang.language as T


# Example 1: Using @tilelang.jit decorator with skip_hardware_check
@tilelang.jit(out_idx=[-1], target="cuda", skip_hardware_check=True)
def simple_add(N, dtype="float32"):
    """Simple element-wise addition kernel."""
    
    @T.prim_func
    def add_kernel(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, 256), threads=256) as (bx,):
            for i in T.serial(256):
                idx = bx * 256 + i
                if idx < N:
                    C[idx] = A[idx] + B[idx]
    
    return add_kernel


# Example 2: Using tilelang.compile directly
def create_kernel_with_compile():
    """Create a kernel using tilelang.compile with skip_hardware_check."""
    
    N = 1024
    
    @T.prim_func
    def multiply_kernel(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, 256), threads=256) as (bx,):
            for i in T.serial(256):
                idx = bx * 256 + i
                if idx < N:
                    C[idx] = A[idx] * B[idx]
    
    # Compile without GPU
    kernel = tilelang.compile(
        multiply_kernel,
        out_idx=[-1],
        target="cuda",
        skip_hardware_check=True
    )
    
    return kernel


# Example 3: Generating source for different targets
def generate_source_for_targets():
    """Generate source code for both CUDA and HIP targets."""
    
    @tilelang.jit(out_idx=[-1], target="cuda", skip_hardware_check=True)
    def kernel_cuda(N):
        @T.prim_func
        def k(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
            with T.Kernel(1, threads=32) as _:
                B[0] = A[0] * 2.0
        return k
    
    @tilelang.jit(out_idx=[-1], target="hip", skip_hardware_check=True)
    def kernel_hip(N):
        @T.prim_func
        def k(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
            with T.Kernel(1, threads=32) as _:
                B[0] = A[0] * 2.0
        return k
    
    cuda_kernel = kernel_cuda(1024)
    hip_kernel = kernel_hip(1024)
    
    return cuda_kernel, hip_kernel


def main():
    print("=" * 80)
    print("Advanced Examples: Compiling Kernels Without GPU")
    print("=" * 80)
    print()
    
    # Example 1: Simple add kernel
    print("Example 1: Simple element-wise add kernel")
    print("-" * 80)
    kernel1 = simple_add(1024)
    source1 = kernel1.get_kernel_source()
    print(f"Generated CUDA source ({len(source1)} characters)")
    print(f"First 200 characters:")
    print(source1[:200] + "...")
    print()
    
    # Example 2: Using compile directly
    print("Example 2: Using tilelang.compile directly")
    print("-" * 80)
    kernel2 = create_kernel_with_compile()
    source2 = kernel2.get_kernel_source()
    print(f"Generated CUDA source ({len(source2)} characters)")
    print(f"First 200 characters:")
    print(source2[:200] + "...")
    print()
    
    # Example 3: Multiple targets
    print("Example 3: Generating source for CUDA and HIP")
    print("-" * 80)
    cuda_k, hip_k = generate_source_for_targets()
    cuda_src = cuda_k.get_kernel_source()
    hip_src = hip_k.get_kernel_source()
    print(f"CUDA source length: {len(cuda_src)} characters")
    print(f"HIP source length: {len(hip_src)} characters")
    print()
    
    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
