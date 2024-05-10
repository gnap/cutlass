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

using namespace cute;

template <typename KernelTraits, class ProblemShape, class Alpha, class Beta,
          class Element = typename KernelTraits::Element,
          class ElementOutput = typename KernelTraits::ElementOutput>
__global__ static __launch_bounds__(decltype(size(typename KernelTraits::TiledMMA{}))::value) void gemm_device(ProblemShape shape_MNK,
                                                                                                               Element const *A,
                                                                                                               Element const *B,
                                                                                                               ElementOutput *C,
                                                                                                               Alpha alpha, Beta beta) {

    using CtaTiler = typename KernelTraits::CtaTiler;
    using TiledMMA = typename KernelTraits::TiledMMA;
    using ASmemLayout = typename KernelTraits::ASmemLayout;
    using BSmemLayout = typename KernelTraits::BSmemLayout;
    using CSmemLayout = typename KernelTraits::CSmemLayout;
    using GmemCopyAB = typename KernelTraits::GmemCopyAB;
    using GmemCopyC = typename KernelTraits::GmemCopyC;
    using SmemCopyAtomAB = typename KernelTraits::SmemCopyAtomAB;
    using SmemCopyAtomC = typename KernelTraits::SmemCopyAtomC;

    CtaTiler cta_tiler;
    TiledMMA tiled_mma;

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

    //
    // Full and Tiled Tensors
    //

    // Represent the full tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK),
                            make_stride(size<2>(shape_MNK), Int<1>{})); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK),
                            make_stride(size<2>(shape_MNK), Int<1>{})); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK),
                            make_stride(size<0>(shape_MNK), Int<1>{})); // (M,N)

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M,BLK_N)

    // Shared memory buffers
    extern __shared__ Element smem_[];
    Element *smemA = smem_;
    Element *smemB = smem_ + cosize_v<ASmemLayout>;

    Tensor sA = make_tensor(make_smem_ptr(smemA), ASmemLayout{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smemB), BSmemLayout{}); // (BLK_N,BLK_K,PIPE)

    //
    // Partition the copying of A and B tiles across the threads
    //
    GmemCopyAB gmem_copy_a, gmem_copy_b;
    ThrCopy thr_copy_a = gmem_copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY,CPY_M,CPY_K,PIPE)

    ThrCopy thr_copy_b = gmem_copy_b.get_slice(threadIdx.x);
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
        copy(gmem_copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe));
        copy(gmem_copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe));
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) {
            ++k_tile_next;
        }
    }

    //
    // Define A/B partitioning and C accumulators
    //

    ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
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
                copy(gmem_copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, smem_pipe_write));
                copy(gmem_copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, smem_pipe_write));
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
            gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
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

    auto r2s_tiled_copy_c = make_tiled_copy_C(SmemCopyAtomC{}, tiled_mma);
    auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(threadIdx.x);
    auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrC);  // (CPY, CPY_M, CPY_N)
    auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC); // (CPY, _1, _1, pipe)

    GmemCopyC s2g_tiled_copy_c;
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
            auto t = make_tensor_like<ElementOutput>(tCrC_r2sx(_, i + j));
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

template <typename Element_, int BLK_M_, int BLK_N_, int BLK_K_, int Stages_ = 3, typename ElementOutput_ = Element_>
struct KernelTraits {

    using Element = Element_;
    using ElementOutput = ElementOutput_;

    using CtaTiler = decltype(make_shape(Int<BLK_M_>{}, Int<BLK_N_>{}, Int<BLK_K_>{}));
    using SmemLayoutAtomAB = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<32>{}),
                    make_stride(Int<32>{}, Int<1>{}))));
    using ASmemLayout = decltype(tile_to_shape(SmemLayoutAtomAB{}, make_shape(Int<BLK_M_>{}, Int<BLK_K_>{}, Int<Stages_>{})));
    using BSmemLayout = decltype(tile_to_shape(SmemLayoutAtomAB{}, make_shape(Int<BLK_N_>{}, Int<BLK_K_>{}, Int<Stages_>{})));

    using SmemLayoutAtomC = decltype(composition(
        Swizzle<2, 3, 3>{}, make_layout(make_shape(Int<32>{}, Int<32>{}),
                                        make_stride(Int<32>{}, Int<1>{}))));
    using CSmemLayout = decltype(tile_to_shape(
        SmemLayoutAtomC{},
        make_shape(Int<32>{}, Int<32>{}, Int<2>{})));

    using TiledMMA = cute::TiledMMA<MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>,
                                    Layout<Shape<_2, _2, _1>>,
                                    Tile<_32, _32, _16>>; // 32x32x16 TiledMMA

    using GmemCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>;
    using GmemCopyAB = decltype(make_tiled_copy(GmemCopyAtom{},
                                                Layout<Shape<_32, _4>, Stride<_4, _1>>{}, // Thr layout 16x8 k-major
                                                Layout<Shape<_1, _8>>{}));                // Val layout  1x8

    using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, ElementOutput>;
    using GmemCopyC =
        decltype(make_tiled_copy(S2GCopyAtomC{},
                                 make_layout(make_shape(Int<32>{}, Int<4>{}),
                                             make_stride(Int<4>{}, Int<1>{})),
                                 make_layout(make_shape(Int<1>{}, Int<8>{}))));

    using Smem_copy_op = SM75_U32x4_LDSM_N;
    using Smem_copy_traits = Copy_Traits<Smem_copy_op>;
    using SmemCopyAtomAB = Copy_Atom<SM75_U32x4_LDSM_N, Element>;

    using SmemCopyAtomC = Copy_Atom<UniversalCopy<int>, ElementOutput>;

    static constexpr int shm_size_AB =
        (cute::cosize_v<ASmemLayout> + cute::cosize_v<BSmemLayout>)*sizeof(Element);
    static constexpr int shm_size_C = cute::cosize_v<CSmemLayout> * sizeof(ElementOutput);

    static constexpr int shm_size =
        cute::max(shm_size_AB, shm_size_C);
};

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
    using kernel_traits = KernelTraits<Element, 128, 128, 32>;

    auto shm_size = kernel_traits::shm_size;

    cudaFuncSetAttribute(gemm_device<kernel_traits, decltype(prob_shape), float, float>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    dim3 dimBlock(size(typename kernel_traits::TiledMMA{}));
    dim3 dimGrid(size(ceil_div(M, 128)),
                 size(ceil_div(N, 128)));
    gemm_device<kernel_traits>
        <<<dimGrid, dimBlock, shm_size, stream>>>(prob_shape, A, B, C, alpha, beta);
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

    int ldA = k, ldB = k, ldC = n;

    /*
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
    */

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
