/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <ParTI/algorithm.hpp>
#include <cassert>
#include <memory>
#include <utility>
#include <vector>
#include <ParTI/error.hpp>
#include <ParTI/errcode.hpp>
#include <ParTI/sptensor.hpp>
#include <ParTI/tensor.hpp>
#include <ParTI/timer.hpp>

#include <thrust/unique.h>

namespace pti {

namespace {


void __global__ ttm_cuda_kernel(
    IndexType const *__restrict__ fiberidx, IndexType const *__restrict__ X_indices_m,
    IndexType nrows, IndexType ncols, IndexType Y_chunk_size, IndexType Y_subchunk_size, IndexType X_chunk_size, IndexType U_stride,
    Scalar *__restrict__ Y_values, Scalar const *__restrict__ X_values, Scalar const *__restrict__ U_values
) {
    IndexType i = blockIdx.x;             // i := mode-n fiber
    IndexType inz_begin = fiberidx[i];    // inz_begin/end := global indices for monde-n fiber of X
    IndexType inz_end = fiberidx[i + 1];
    IndexType r = threadIdx.x;
    for(IndexType k = threadIdx.y; k < Y_subchunk_size; k += blockDim.y) {
        Scalar accumulate = 0;
        for(IndexType j = inz_begin; j < inz_end; ++j) { // loop over fiber i
            IndexType c = X_indices_m[j]; // get mode-n index of X: c ∈ [1, size(mode-n)]
            if(r < nrows && c < ncols) {
                accumulate += X_values[j * X_chunk_size + k] * U_values[r * U_stride + c];
            }
        }
        Y_values[i * Y_chunk_size + r * Y_subchunk_size + k] += accumulate;
    }
}

/* impl_num = 15 */
__global__ void spt_TTMRankRBNnzKernelSM(
    Scalar *Y_val, 
    IndexType Y_stride, IndexType Y_nnz,
    const Scalar * __restrict__ X_val, 
    IndexType X_nnz, 
    const IndexType * __restrict__ X_inds_m,
    const IndexType * __restrict__ fiberidx_val, 
    const Scalar * __restrict__ U_val, 
    IndexType U_nrows, 
    IndexType U_ncols, 
    IndexType U_stride) 
{
    extern __shared__ Scalar mem_pool[];
    Scalar * const Y_shr = (Scalar *) mem_pool; // size U_ncols

    IndexType num_loops_nnz = 1;
    IndexType const nnz_per_loop = gridDim.x * blockDim.y;
    if(Y_nnz > nnz_per_loop) {
        num_loops_nnz = (Y_nnz + nnz_per_loop - 1) / nnz_per_loop;
    }
     
    // Global indices of Y: Fiber = x and Inner fiber = r
    // Local indices: tidx and tidy
    const IndexType tidx = threadIdx.x;
    const IndexType tidy = threadIdx.y;
    IndexType x;
    const IndexType num_loops_r = U_ncols / blockDim.x;
    const IndexType rest_loop = U_ncols - num_loops_r * blockDim.x;
    IndexType r; // column idx of U


    for(IndexType l=0; l<num_loops_r; ++l) { // blockDim.x parallelised over cols(U)
        r = tidx + l * blockDim.x; // r: column idx of U
        for(IndexType nl=0; nl<num_loops_nnz; ++nl) { // Grid strided-pattern?
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;

            Y_shr[tidy * blockDim.x + tidx] = 0;
            __syncthreads();

            if(x < Y_nnz) { // Why is this not above at line 348
                const IndexType inz_begin = fiberidx_val[x];
                const IndexType inz_end = fiberidx_val[x+1];
                for(IndexType i = inz_begin; i < inz_end; ++i) { // loop over a n-fiber
                    const IndexType row = X_inds_m[i]; // row of U
                    // Loop over nnz in n-fiber of X and multiply with corresponding
                    // U col elements and accumulate in single element of Y
                    Y_shr[tidy*blockDim.x + tidx] += X_val[i] * U_val[row*U_stride + r];  // Original
                    // Y_shr[tidy*blockDim.x + tidx] += X_val[i] * U_val[r*U_stride + row]; 
                }
                __syncthreads();

                Y_val[x*Y_stride + r] = Y_shr[tidy*blockDim.x + tidx];
                __syncthreads();
            }
        }
    }


    if(rest_loop > 0 && tidx < rest_loop) {
        r = tidx + num_loops_r * blockDim.x;

        for(IndexType nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;

            Y_shr[tidy * blockDim.x + tidx] = 0;
            __syncthreads();

            if(x < Y_nnz) {
                const IndexType inz_begin = fiberidx_val[x];
                const IndexType inz_end = fiberidx_val[x+1];
                for(IndexType i = inz_begin; i < inz_end; ++i) {
                    const IndexType row = X_inds_m[i];
                    Y_shr[tidy*blockDim.x + tidx] += X_val[i] * U_val[row*U_stride + r];  // Original
                    // Y_shr[tidy*blockDim.x + tidx] += X_val[i] * U_val[r*U_stride + row]; 
                }
                __syncthreads();

                Y_val[x*Y_stride + r] = Y_shr[tidy*blockDim.x + tidx];
                __syncthreads();
            }
        }
    }

}

}


SparseTensor tensor_times_matrix_cuda(SparseTensor& X, Tensor& U, IndexType mode, CudaDevice* cuda_dev, bool skip_sort) {
    IndexType nmodes = X.nmodes;
    IndexType nspmodes = X.sparse_order.size();

    ptiCheckError(mode >= nmodes, ERR_SHAPE_MISMATCH, "mode >= X.nmodes");
    ptiCheckError(X.is_dense(cpu)[mode], ERR_UNKNOWN, "X.is_dense[mode] != false");

    ptiCheckError(U.nmodes != 2, ERR_SHAPE_MISMATCH, "U.nmodes != 2");
    ptiCheckError(U.storage_order(cpu)[0] != 0, ERR_SHAPE_MISMATCH, "U.storage_order[0] != 0");
    ptiCheckError(U.storage_order(cpu)[1] != 1, ERR_SHAPE_MISMATCH, "U.storage_order[1] != 1");

    IndexType nrows = U.shape(cpu)[0];
    IndexType ncols = U.shape(cpu)[1];
    IndexType Ustride = U.strides(cpu)[1];

    ptiCheckError(X.shape(cpu)[mode] != nrows, ERR_SHAPE_MISMATCH, "X.shape[mode] != U.ncols");

    if(skip_sort) {
        ptiCheckError(X.sparse_order(cpu)[nspmodes - 1] != mode, ERR_SHAPE_MISMATCH, "X.sparse_order[-1] != mode");
    } else {
        Timer timer_sort(cpu);
        timer_sort.start();

        std::unique_ptr<IndexType[]> sort_order(new IndexType [nspmodes]);
        for(IndexType m = 0, i = 0; m < nspmodes; ++m) {
            IndexType sort_order_mode = X.sparse_order(cpu)[m];
            if(sort_order_mode != mode) {
                sort_order[i] = sort_order_mode;
                ++i;
            }
        }
        sort_order[nspmodes - 1] = mode;
        X.sort_index(sort_order.get());

        timer_sort.stop();
        timer_sort.print_elapsed_time("CUDA TTM Sort");
    }
    for (int i = 0; i < X.num_chunks * X.chunk_size; ++i) {
        std::cout << X.modes_d[0][i] << ' ' << X.modes_d[1][i] << ' ' << X.modes_d[2][i] << ' ' << X.values_thrust_d[i] << '\n';
    }
    Timer timer_thrust(cpu);
    timer_thrust.start();
    X.sort_thrust(true, mode);
    timer_thrust.stop();
    timer_thrust.print_elapsed_time("THRUST SORT");
    std::printf("X = %s\n", X.to_string(1, 10).c_str());

    std::unique_ptr<IndexType[]> Y_shape(new IndexType [nmodes]);
    for(IndexType m = 0; m < nmodes; ++m) {
        if(m != mode) {
            Y_shape[m] = X.shape(cpu)[m];
        } else {
            Y_shape[m] = ncols;
        }
    }
    bool const* X_is_dense = X.is_dense(cpu);
    std::unique_ptr<bool[]> Y_is_dense(new bool [nmodes]);
    for(IndexType m = 0; m < nmodes; ++m) {
        Y_is_dense[m] = X_is_dense[m] || m == mode;
    }

    SparseTensor Y(nmodes, Y_shape.get(), Y_is_dense.get());
    IndexType* X_dense_order = X.dense_order(cpu);
    IndexType* Y_dense_order = Y.dense_order(cpu);
    for(IndexType m = 0; m < Y.dense_order.size() - 1; ++m) {
        Y_dense_order[m] = X_dense_order[m];
    }
    Y_dense_order[Y.dense_order.size() - 1] = mode;
    Y.sort_index(X.sparse_order(cpu));

    Timer timer_setidx(cpu);
    timer_setidx.start();

    std::vector<IndexType> fiberidx;
    set_semisparse_indices_by_sparse_ref(Y, fiberidx, X, mode);
    for (const auto& e : fiberidx) std::cout << e << ' ';
    std::cout << '\n';

    thrust::device_vector<IndexType> contracted_mode(X.modes_d[mode]);

    thrust::device_vector<IndexType> fiberidx_thrust(X.chunk_size * X.num_chunks);
    set_semisparse_indices_by_sparse_ref_thrust(Y, fiberidx_thrust, X, mode);

    for (int i = 0; i < X.modes_d[0].size(); ++i) {
        std::cout << X.modes_d[0][i] << ' ' << X.modes_d[1][i] << ' ' << X.modes_d[2][i] << '\n';
    }

    std::cout << "Mode0:\n";
    for (const auto& e : contracted_mode) std::cout << e << ' ';
    std::cout << '\n';


    timer_setidx.stop();
    timer_setidx.print_elapsed_time("CUDA TTM SetIdx");
    // std::printf("Y = %s\n", Y.to_string(1, 10).c_str());
    // for (auto& e : Y.indices_thrust_h) {
    //     std::cout << e.x << ' ' << e.y << ' ' << e.z << '\n';
    // }
    // std::cout << '\n';
    // printf("Fiberidx length = %zu\n", fiberidx.size());
    // for (const auto& e : fiberidx) std::cout << e << ' ';
    // std::cout << '\n';

    // Scalar* X_values = X.values(cuda_dev->mem_node);
    Scalar* Y_values = Y.values(cuda_dev->mem_node);
    // Scalar* U_values = U.values(cuda_dev->mem_node);
    // IndexType* X_indices_m = X.indices[mode](cuda_dev->mem_node);
    // IndexType *dev_fiberidx = (IndexType *) session.mem_nodes[cuda_dev->mem_node]->malloc(fiberidx.size() * sizeof (IndexType));
    // session.mem_nodes[cuda_dev->mem_node]->memcpy_from(dev_fiberidx, fiberidx.data(), *session.mem_nodes[cpu], fiberidx.size() * sizeof (IndexType));


    Scalar* X_values = thrust::raw_pointer_cast(&X.values_thrust_d[0]);
    // Scalar* Y_values = thrust::raw_pointer_cast(&Y.values_thrust_d[0]);
    Scalar* U_values = U.values(cuda_dev->mem_node);
    IndexType* X_indices_m = thrust::raw_pointer_cast(&contracted_mode[0]);
    IndexType *dev_fiberidx = thrust::raw_pointer_cast(&fiberidx_thrust[0]);


    IndexType Y_subchunk_size = X.chunk_size;
    IndexType Y_num_subchunks = Y.strides(cpu)[mode];
    assert(Y_num_subchunks * Y_subchunk_size == Y.chunk_size);

    const IndexType max_nblocks = 32768;
    const IndexType max_nthreads_per_block = 256;
    IndexType max_nthreadsy = 32;

    IndexType nthreadsx = 1;
    IndexType nthreadsy = 1;
    IndexType all_nblocks = 0;
    IndexType nblocks = 0;
    IndexType shmen_size = 0;

    if(ncols <= max_nthreadsy)
        nthreadsx = ncols;
    else
        nthreadsx = max_nthreadsy;
    nthreadsy = max_nthreads_per_block / nthreadsx;

    IndexType Y_nnz = Y.num_chunks;
    if(Y_nnz < nthreadsy) {
        nthreadsy = Y_nnz;
        nblocks = 1;
    } else {
        all_nblocks = (Y_nnz + nthreadsy -1) / nthreadsy;
        if(all_nblocks < max_nblocks) {
            nblocks = all_nblocks;
        } else {
            nblocks = max_nblocks;
        }
    }
    shmen_size = nthreadsx * nthreadsy * sizeof(Scalar);
    assert(shmen_size >= nthreadsx * nthreadsy * sizeof (Scalar));
    dim3 dimBlock(nthreadsx, nthreadsy);
    printf("SHMEM size: %lu  (%lu bytes)\n", shmen_size / sizeof(Scalar), shmen_size);
    printf("all_nblocks: %lu, nthreadsx: %lu, nthreadsy: %lu\n", all_nblocks, nthreadsx, nthreadsy);

    printf("X_nnz: %lu\n", X.num_chunks);
    printf("U_rows: %lu; U_cols: %lu; U_stride: %lu\n", nrows, ncols, Ustride);
    printf("Y_nnz: %lu; Y_stride: %lu\n", Y.num_chunks, Y.chunk_size);




    Timer timer_kernel(cuda_dev->device_id);
    timer_kernel.start();
    IndexType kernel_blockDim_y = std::min(Y_subchunk_size, 1024 / Y_num_subchunks);
    assert(kernel_blockDim_y > 0);
    // std::fprintf(stderr, "[CUDA TTM Kernel] Launch ttm_cuda_kernel<<<%zu, (%zu, %zu), 0>>()\n", Y.num_chunks, Y_num_subchunks, kernel_blockDim_y);
    // ttm_cuda_kernel<<<Y.num_chunks, dim3(Y_num_subchunks, kernel_blockDim_y), 0>>>(dev_fiberidx, X_indices_m, nrows, ncols, Y.chunk_size, Y_subchunk_size, X.chunk_size, Ustride, Y_values, X_values, U_values);

    spt_TTMRankRBNnzKernelSM<<<nblocks, dimBlock, shmen_size>>>(
        Y_values, Y.chunk_size, Y.num_chunks,
        X_values, X.num_chunks, X_indices_m,
        dev_fiberidx, U_values, nrows, ncols, Ustride);

    int result = cudaDeviceSynchronize();
    timer_kernel.stop();
    timer_kernel.print_elapsed_time("CUDA TTM Kernel");
    ptiCheckCUDAError(result != 0);

    // session.mem_nodes[cuda_dev->mem_node]->free(dev_fiberidx);

    return Y;
}

}
