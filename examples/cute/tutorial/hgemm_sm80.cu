#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "cublaslt-gemm.h"

template <class ProblemShape, class CtaTiler, class Element,
          class AStride, class ASmemLayout, class TiledCopyA, class SmemCopyAtomA,
          class BStride, class BSmemLayout, class TiledCopyB, class SmemCopyAtomB,
          class TC, class CStride, class CSmemLayout, class TiledCopyC, class SmemCopyAtomC,
          class TiledMma, class Alpha, class Beta>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value) void gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                                                                                        Element const *A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, SmemCopyAtomA smem_copy_atom_a,
                                                                                        Element const *B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, SmemCopyAtomB smem_copy_atom_b,
                                                                                        TC *C, CStride dC, CSmemLayout sC_layout, TiledCopyC copy_c, SmemCopyAtomC smem_copy_atom_c, TiledMma mma,
                                                                                        Alpha alpha, Beta beta) {
    using namespace cute;

    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{}); // (M, N, K)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{}); // (BLK_M, BLK_N, BLK_K)

    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<BSmemLayout>::value);
    static_assert(is_static<CSmemLayout>::value);

    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler)); // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler)); // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler)); // BLK_K

    CUTE_STATIC_ASSERT_V(congruent(select<0, 2>(shape_MNK), dA)); // dA strides for shape MK
    CUTE_STATIC_ASSERT_V(congruent(select<1, 2>(shape_MNK), dB)); // dB strides for shape NK
    CUTE_STATIC_ASSERT_V(congruent(select<0, 1>(shape_MNK), dC)); // dC strides for shape MN

    //
    // Full and Tiled Tensors
    //

    // Represent the full tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // (M,N)

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M,BLK_N)

    // Shared memory buffers
    __shared__ Element smemA[cosize_v<ASmemLayout>];
    __shared__ Element smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N,BLK_K,PIPE)

    //
    // Partition the copying of A and B tiles across the threads
    //
    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY,CPY_M,CPY_K,PIPE)

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY,CPY_N,CPY_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA)); // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA)); // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB)); // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB)); // CPY_K

    //
    // PREFETCH
    //

    auto K_PIPE_MAX = size<3>(tAsA);

    // Total count of tiles
    int k_tile_count = size<3>(tAgA);
    // Current tile index in gmem to read from
    int k_tile_next = 0;

    // Start async loads for all pipes but the last
    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
        copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe));
        copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe));
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) {
            ++k_tile_next;
        }
    }

    //
    // Define A/B partitioning and C accumulators
    //

    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB); // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCgC = thr_mma.partition_C(gC); // (MMA,MMA_M,MMA_N)

    // Allocate registers for pipelining
    Tensor tCrA = thr_mma.make_fragment_A(tCsA(_, _, _, 0)); // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB(_, _, _, 0)); // (MMA,MMA_N,MMA_K)
    // Allocate the accumulators -- same size as the projected data
    Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA,MMA_M,MMA_N)

    CUTE_STATIC_ASSERT_V(shape(tCrA) == shape(tCsA));     // (MMA,MMA_M,MMA_K)
    CUTE_STATIC_ASSERT_V(shape(tCrB) == shape(tCsB));     // (MMA,MMA_N,MMA_K)
    CUTE_STATIC_ASSERT_V(shape(tCrC) == shape(tCgC));     // (MMA,MMA_M,MMA_N)
    CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCsA)); // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCsB)); // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB)); // MMA_K

    // Clear the accumulators
    clear(tCrC);

#if 0
  if(thread0()) {
    print("  mA : "); print(  mA); print("\n");
    print("  gA : "); print(  gA); print("\n");
    print("  sA : "); print(  sA); print("\n");
    print("tAgA : "); print(tAgA); print("\n");
    print("tAsA : "); print(tAsA); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mB : "); print(  mB); print("\n");
    print("  gB : "); print(  gB); print("\n");
    print("  sB : "); print(  sB); print("\n");
    print("tBgB : "); print(tBgB); print("\n");
    print("tBsB : "); print(tBsB); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mC : "); print(  mC); print("\n");
    print("  gC : "); print(  gC); print("\n");
    print("tCsA : "); print(tCsA); print("\n");
    print("tCsB : "); print(tCsB); print("\n");
    print("tCgC : "); print(tCgC); print("\n");
    print("tCrA : "); print(tCrA); print("\n");
    print("tCrB : "); print(tCrB); print("\n");
    print("tCrC : "); print(tCrC); print("\n");
  }
#endif

#if 1

    // Current pipe index in smem to read from
    int smem_pipe_read = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = K_PIPE_MAX - 1;

    // Pipe slice
    Tensor tCsA_p = tCsA(_, _, _, smem_pipe_read);
    Tensor tCsB_p = tCsB(_, _, _, smem_pipe_read);

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA);

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
        // Wait until our first prefetched tile is loaded in
        cp_async_wait<K_PIPE_MAX - 2>();
        __syncthreads();

        // Prefetch the first rmem from the first k-tile
        copy(tCsA_p(_, _, Int<0>{}), tCrA(_, _, Int<0>{}));
        copy(tCsB_p(_, _, Int<0>{}), tCrB(_, _, Int<0>{}));
    }

    //
    // PIPELINED MAIN LOOP
    // TUTORIAL: Example of a gemm loop that pipelines shared memory using SM80's cp.async instructions
    //           and explicit pipelines in shared memory.
    //   Data is read from global(k_tile_next) to shared(smem_pipe_write).
    //   Data is read from shared(smem_pipe_read) to registers(k_block_next).
    //   Data is computed on registers(b_block).
    //
    //   This allows all copies and compute to overlap:
    //     Copy from gmem->smem can overlap with copies from smem->rmem and compute on rmem.
    //     Copy from smem->rmem can overlap with compute on rmem.
    //

    CUTE_NO_UNROLL
    while (k_tile_count > -(K_PIPE_MAX - 1)) {
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            if (k_block == K_BLOCK_MAX - 1) {
                // Slice the smem_pipe_read smem
                tCsA_p = tCsA(_, _, _, smem_pipe_read);
                tCsB_p = tCsB(_, _, _, smem_pipe_read);

                // Commit the smem for smem_pipe_read
                cp_async_wait<K_PIPE_MAX - 2>();
                __syncthreads();
            }

            // Load A, B shmem->regs for k_block+1
            auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX; // static
            copy(tCsA_p(_, _, k_block_next), tCrA(_, _, k_block_next));
            copy(tCsB_p(_, _, k_block_next), tCrB(_, _, k_block_next));
            // Copy gmem to smem before computing gemm on each k-pipe
            if (k_block == 0) {
                copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, smem_pipe_write));
                copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, smem_pipe_write));
                cp_async_fence();

                // Advance the gmem tile
                --k_tile_count;
                if (k_tile_count > 0) {
                    ++k_tile_next;
                }

                // Advance the smem pipe
                smem_pipe_write = smem_pipe_read;
                ++smem_pipe_read;
                smem_pipe_read = (smem_pipe_read == K_PIPE_MAX) ? 0 : smem_pipe_read;
            }
            // Thread-level register gemm for k_block
            gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }
    }

#endif

    //
    // Epilogue
    //

    // axpby(alpha, tCrC, beta, tCgC);
    //  use less shared memory as a scratchpad tile to use large wide instuction
    //  Dreg -> shm -> reg -> global
    axpby(alpha, tCrC, beta, tCrC);
    auto sC = make_tensor(sA(_, _, smem_pipe_read).data(), CSmemLayout{});

    auto r2s_tiled_copy_c = make_tiled_copy_C(SmemCopyAtomC{}, mma);
    auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(threadIdx.x);
    auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrC);  // (CPY, CPY_M, CPY_N)
    auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC); // (CPY, _1, _1, pipe)

    TiledCopyC s2g_tiled_copy_c;
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(threadIdx.x);
    auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC); // (CPY, _1, _1, pipe)
    auto tCgC_s2g = s2g_thr_copy_c.partition_D(gC); // (CPY, CPY_M, CPY_N)

    auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g); // (CPY_, CPY_MN)
    auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s); // (CPY_, CPY_MN)

#if 0
    if (thread0()) {
        print(tCrC_r2s);
        print(tCsC_r2s);
        print("\n");
    }
#endif
    int step = size<3>(tCsC_r2s); // pipe
    CUTE_UNROLL
    for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
        // reg -> shm
        CUTE_UNROLL
        for (int j = 0; j < step; ++j) {
            // we add a temp tensor to cope with accumulator and output data type
            // difference
            auto t = make_tensor_like<TC>(tCrC_r2sx(_, i + j));
            cute::copy(tCrC_r2sx(_, i + j), t);

            cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
        }
        __syncthreads();

        CUTE_UNROLL
        // shm -> global
        for (int j = 0; j < step; ++j) {

            cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
        }

        __syncthreads();
    }
}

// Setup params for a TN GEMM
template <class Element, class TC,
          class Alpha, class Beta>
void gemm_tn(int m, int n, int k,
             Alpha alpha,
             Element const *A, int ldA,
             Element const *B, int ldB,
             Beta beta,
             TC *C, int ldC,
             cudaStream_t stream = 0) {
    using namespace cute;

    // Define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K); // (M, N, K)

    // Define TN strides (mixed)
    auto dA = make_stride(ldA, Int<1>{}); // (dM, dK)
    auto dB = make_stride(ldB, Int<1>{}); // (dN, dK)
    // auto dC = make_stride(Int<1>{}, ldC); // (dM, dN)
    auto dC = make_stride(ldC, Int<1>{}); // (dM, dN)

    // Define CTA tile sizes (static)
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<32>{};
    auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)
    auto bP = Int<3>{};                      // Pipeline

    // Define the smem layouts (static)
    auto SmemLayoutAtom = composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<32>{}),
                    make_stride(Int<32>{}, Int<1>{})));
    auto sA = tile_to_shape(SmemLayoutAtom, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(SmemLayoutAtom, make_shape(bN, bK, bP));

    using SmemLayoutAtomC = decltype(composition(
        Swizzle<2, 3, 3>{}, make_layout(make_shape(Int<32>{}, Int<32>{}),
                                        make_stride(Int<32>{}, Int<1>{}))));
    auto sC = tile_to_shape(
        SmemLayoutAtomC{},
        make_shape(Int<32>{}, Int<32>{}, Int<2>{}));

    // Define the thread layouts (static)

    TiledMMA mmaC = make_tiled_mma(MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{},
                                   Layout<Shape<_2, _2, _1>>{},
                                   Tile<_32, _32, _16>{}); // 32x32x16 TiledMMA

    auto gmem_copy_atom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{};
    TiledCopy copyA = make_tiled_copy(gmem_copy_atom,
                                      Layout<Shape<_32, _4>, Stride<_4, _1>>{}, // Thr layout 16x8 k-major
                                      Layout<Shape<_1, _8>>{});                 // Val layout  1x8
    TiledCopy copyB = copyA;

    using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, TC>;
    using copyC =
        decltype(make_tiled_copy(S2GCopyAtomC{},
                                 make_layout(make_shape(Int<32>{}, Int<4>{}),
                                             make_stride(Int<4>{}, Int<1>{})),
                                 make_layout(make_shape(Int<1>{}, Int<8>{}))));

    using Smem_copy_op = SM75_U32x4_LDSM_N;
    using Smem_copy_traits = Copy_Traits<Smem_copy_op>;
    auto smem_copy_atom = Copy_Atom<Smem_copy_traits, Element>{};

    auto smem_copy_atom_c = Copy_Atom<UniversalCopy<int>, TC>{};

#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

    dim3 dimBlock(size(mmaC));
    dim3 dimGrid(size(ceil_div(M, bM)),
                 size(ceil_div(N, bN)));
    gemm_device<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler,
                                                  A, dA, sA, copyA, smem_copy_atom,
                                                  B, dB, sB, copyB, smem_copy_atom,
                                                  C, dC, sC, copyC{}, smem_copy_atom_c, mmaC,
                                                  alpha, beta);
}

template <class Element, class TC,
          class Alpha, class Beta>
void gemm(char transA, char transB, int m, int n, int k,
          Alpha alpha,
          Element const *A, int ldA,
          Element const *B, int ldB,
          Beta beta,
          TC *C, int ldC,
          cudaStream_t stream = 0) {
    if (transA == 'T' && transB == 'N') {
        return gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
    }
    assert(false && "Not implemented");
}

int main(int argc, char **argv) {
    cudaDeviceProp props;
    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    if (props.major < 8) {
        std::cout << "This example requires an Ampere GPU or newer (CC >= 80)" << std::endl;
        // Return 0 so tests pass if run on unsupported architectures or CUDA Toolkits.
        return 0;
    }

    int m = 5120;
    if (argc >= 2)
        sscanf(argv[1], "%d", &m);

    int n = 5120;
    if (argc >= 3)
        sscanf(argv[2], "%d", &n);

    int k = 4096;
    if (argc >= 4)
        sscanf(argv[3], "%d", &k);

    char transA = 'T';
    if (argc >= 5)
        sscanf(argv[4], "%c", &transA);

    char transB = 'N';
    if (argc >= 6)
        sscanf(argv[5], "%c", &transB);

    using Element = cute::half_t;
    using TC = cute::half_t;
    using TI = float;

    TI alpha = 1.0;
    TI beta = 0.0;

    std::cout << "M = " << m << std::endl;
    std::cout << "N = " << n << std::endl;
    std::cout << "K = " << k << std::endl;
    std::cout << "C = A^" << transA << " B^" << transB << std::endl;

    thrust::host_vector<Element> h_A(m * k);
    thrust::host_vector<Element> h_B(n * k);
    thrust::host_vector<TC> h_C(m * n);
    thrust::host_vector<TC> h_Cblas(m * n);

    for (int j = 0; j < m * k; ++j)
        h_A[j] = static_cast<Element>(2 * (rand() / double(RAND_MAX)) - 1);
    for (int j = 0; j < n * k; ++j)
        h_B[j] = static_cast<Element>(2 * (rand() / double(RAND_MAX)) - 1);
    for (int j = 0; j < m * n; ++j)
        h_C[j] = static_cast<TC>(-1);

    for (int j = 0; j < m * n; ++j)
        h_Cblas[j] = static_cast<TC>(-1);

    thrust::device_vector<Element> d_A = h_A;
    thrust::device_vector<Element> d_B = h_B;
    thrust::device_vector<TC> d_Cblas = h_Cblas;
    thrust::device_vector<TC> d_C = h_C;
    CUTE_CHECK_LAST();

    double gflops = (2.0 * m * n * k) * 1e-9;

    const int timing_iterations = 1;
    GPU_Clock timer;

    int ldA = 0, ldB = 0, ldC = m;

    if (transA == 'N') {
        ldA = m;
    } else if (transA == 'T') {
        ldA = k;
    } else {
        assert(false);
    }

    if (transB == 'N') {
        ldB = k;
    } else if (transB == 'T') {
        ldB = n;
    } else {
        assert(false);
    }

    // Run once
    d_C = h_C;
    for (int i = 0; i < 5; ++i) {
        gemm(transA, transB, m, n, k,
             alpha,
             d_A.data().get(), ldA,
             d_B.data().get(), ldB,
             beta,
             d_C.data().get(), ldC);
    }
    CUTE_CHECK_LAST();
    thrust::host_vector<TC> cute_result = d_C;

    CublasLtGemm<Element, TC> cublaslt_gemm;
    cublaslt_gemm.init(d_Cblas.data().get(), d_B.data().get(), d_A.data().get(), n, m, k);
    cublaslt_gemm.run();
    thrust::host_vector<TC> cublas_result = d_Cblas;
    cudaDeviceSynchronize();

    const auto cute_result_ptr = cute_result.data();
    auto cublas_result_ptr = cublas_result.data();
    auto tD_host = cute::make_tensor(cute_result_ptr, cute::make_shape(m, n), cute::make_stride(n, 1));
    auto tD_host_cublaslt =
        cute::make_tensor(cublas_result_ptr, cute::make_shape(m, n), cute::make_stride(n, 1));

    auto tile = cute::make_tile(min(8, m), min(8, n));
    auto t32x32 = cute::local_tile(tD_host, tile, cute::make_coord(0, 0));
    auto t32x32_cublaslt = cute::local_tile(tD_host_cublaslt, tile, cute::make_coord(0, 0));

    print_tensor(t32x32);
    print_tensor(t32x32_cublaslt);

    for (int i = 0; i < timing_iterations; ++i) {

        // Timing iterations
        timer.start();
        gemm(transA, transB, m, n, k,
             alpha,
             d_A.data().get(), ldA,
             d_B.data().get(), ldB,
             beta,
             d_C.data().get(), ldC);
        double cute_time = timer.milliseconds() / float(1);
        CUTE_CHECK_LAST();
        printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time * 1e3, cute_time);
    }

    return 0;
}
