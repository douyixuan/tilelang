#!/usr/bin/env python3
"""
Test script to verify that kernels can be compiled to source code 
without requiring GPU hardware.

This demonstrates the fix for issue #1049.
"""

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1], target="cuda", skip_hardware_check=True)
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    """Simple matrix multiplication kernel."""

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


def main():
    print("=" * 80)
    print("Testing compilation without GPU hardware (Issue #1049)")
    print("=" * 80)
    print()
    
    try:
        # Compile the kernel without GPU
        print("Compiling matmul kernel with skip_hardware_check=True...")
        kernel = matmul(1024, 1024, 1024, 128, 128, 32)
        print("✓ Kernel compiled successfully!")
        print()
        
        # Get the generated source code
        print("Generated CUDA source code:")
        print("-" * 80)
        source = kernel.get_kernel_source()
        # Print first 50 lines of source
        lines = source.split('\n')
        for i, line in enumerate(lines[:50], 1):
            print(f"{i:3}: {line}")
        if len(lines) > 50:
            print(f"... ({len(lines) - 50} more lines)")
        print("-" * 80)
        print()
        
        print("✓ Source code generated successfully!")
        print()
        print("=" * 80)
        print("Test passed! Kernels can now be compiled without GPU hardware.")
        print("=" * 80)
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
