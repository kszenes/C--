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
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>
#include <ParTI/error.hpp>
#include <ParTI/errcode.hpp>
#include <ParTI/sptensor.hpp>
#include <ParTI/tensor.hpp>
#include <ParTI/timer.hpp>
#include <ParTI/utils.hpp>
#include <iostream>

namespace pti {

SparseTensor tensor_times_matrix(SparseTensor& X, Tensor& U, size_t mode, Device *device, bool skip_sort) {
    if(CudaDevice *cuda_device = dynamic_cast<CudaDevice *>(device)) {
#ifdef PARTI_USE_CUDA
        return tensor_times_matrix_cuda(X, U, mode, cuda_device, skip_sort);
#else
        (void) cuda_device;
        ptiCheckError(true, ERR_BUILD_CONFIG, "CUDA not enabled");
#endif
    }

    if(const char *env_pti_use_omp_ttm = std::getenv("PTI_USE_OMP_TTM")) {
        if(env_pti_use_omp_ttm != nullptr && std::strcmp(env_pti_use_omp_ttm, "1") == 0) {
            return tensor_times_matrix_omp(X, U, mode, skip_sort);
        }
    }

    size_t nmodes = X.nmodes;
    size_t nspmodes = X.sparse_order.size();
    std::cout << "Contracting mode: " << mode << '\n';

    ptiCheckError(mode >= nmodes, ERR_SHAPE_MISMATCH, "mode >= X.nmodes");
    ptiCheckError(X.is_dense(cpu)[mode], ERR_UNKNOWN, "X.is_dense[mode] != false");

    ptiCheckError(U.nmodes != 2, ERR_SHAPE_MISMATCH, "U.nmodes != 2");
    ptiCheckError(U.storage_order(cpu)[0] != 0, ERR_SHAPE_MISMATCH, "U.storage_order[0] != 0");
    ptiCheckError(U.storage_order(cpu)[1] != 1, ERR_SHAPE_MISMATCH, "U.storage_order[1] != 1");

    size_t nrows = U.shape(cpu)[0];
    size_t ncols = U.shape(cpu)[1];
    size_t Ustride = U.strides(cpu)[1];

    ptiCheckError(X.shape(cpu)[mode] != ncols, ERR_SHAPE_MISMATCH, "X.shape[mode] != U.ncols");

    if(skip_sort) {
        ptiCheckError(X.sparse_order(cpu)[nspmodes - 1] != mode, ERR_SHAPE_MISMATCH, "X.sparse_order[-1] != mode");
    } else {
        Timer timer_sort(cpu);
        timer_sort.start();

        std::unique_ptr<size_t[]> sort_order(new size_t [nspmodes]);
        for(size_t m = 0, i = 0; m < nspmodes; ++m) {
            size_t sort_order_mode = X.sparse_order(cpu)[m];
            if(sort_order_mode != mode) {
                sort_order[i] = sort_order_mode;
                ++i;
            }
        }
        sort_order[nspmodes - 1] = mode;
        X.sort_index(sort_order.get());

        timer_sort.stop();
        std::printf("X = %s\n", X.to_string(true).c_str());
        timer_sort.print_elapsed_time("CPU TTM Sort");
    }

    std::unique_ptr<size_t[]> Y_shape(new size_t [nmodes]);
    for(size_t m = 0; m < nmodes; ++m) {
        if(m != mode) {
            Y_shape[m] = X.shape(cpu)[m];
        } else {
            Y_shape[m] = nrows;
        }
    }
    bool const* X_is_dense = X.is_dense(cpu);
    std::unique_ptr<bool[]> Y_is_dense(new bool [nmodes]);
    for(size_t m = 0; m < nmodes; ++m) {
        Y_is_dense[m] = X_is_dense[m] || m == mode;
    }

    SparseTensor Y(nmodes, Y_shape.get(), Y_is_dense.get());
    size_t* X_dense_order = X.dense_order(cpu);
    size_t* Y_dense_order = Y.dense_order(cpu);
    for(size_t m = 0; m < Y.dense_order.size() - 1; ++m) {
        Y_dense_order[m] = X_dense_order[m];
    }
    Y_dense_order[Y.dense_order.size() - 1] = mode;
    Y.sort_index(X.sparse_order(cpu));

    Timer timer_setidx(cpu);
    timer_setidx.start();

    std::vector<size_t> fiberidx;
    set_semisparse_indices_by_sparse_ref(Y, fiberidx, X, mode);
    // set_semisparse_indices_by_sparse_ref_scan_seq(Y, fiberidx, X, mode);
    // set_semisparse_indices_by_sparse_ref_scan_omp_task(Y, fiberidx, X, mode);
    // set_semisparse_indices_by_sparse_ref_scan_omp(Y, fiberidx, X, mode);
    // printf("fiberidx: \n");
    // for(size_t i = 0; i < fiberidx.size(); ++i) {
    //     printf("%zu ", fiberidx[i]);
    // }
    // printf("\n"); fflush(stdout);

    timer_setidx.stop();
    // std::cout << "fiberidx: ";
    // for (const auto& e : fiberidx) {
    //     std::cout << e << ' ';
    // }
    std::cout << '\n';
    timer_setidx.print_elapsed_time("CPU TTM SetIdx");

    Scalar* X_values = X.values(cpu);
    Scalar* Y_values = Y.values(cpu);
    Scalar* U_values = U.values(cpu);
    size_t* X_indices_m = X.indices[mode](cpu);

    /*
    std::unique_ptr<size_t[]> idxY(new size_t[nmodes]);
    std::unique_ptr<size_t[]> idxX(new size_t[nmodes]);
    std::unique_ptr<size_t[]> idxU(new size_t[nmodes]);
    */

    size_t Y_subchunk_size = X.chunk_size;
    size_t Y_num_subchunks = Y.strides(cpu)[mode];
    assert(Y_num_subchunks * Y_subchunk_size == Y.chunk_size);


    // i is chunk-level on Y
    std::cout << "=== Chunks ===\n";
    std::cout << "X.num_chunks = " << X.num_chunks << '\n';
    std::cout << "X.chunk_size = " << X.chunk_size << '\n';
    std::cout << "Y.num_chunks = " << Y.num_chunks << '\n';
    std::cout << "Y.chunk_size = " << Y.chunk_size << '\n';
    std::cout << "Y_num_subchunks = " << Y_num_subchunks << '\n';
    std::cout << "Y_subchunk_size = " << Y_subchunk_size << '\n';
    std::cout << "Ustride = " << Ustride << '\n';
    Timer timer_kernel(cpu);
    timer_kernel.start();
    for(size_t i = 0; i < Y.num_chunks; ++i) {
        size_t inz_begin = fiberidx[i];
        size_t inz_end = fiberidx[i + 1];
        // j is chunk-level on X, j := global (vectorized) index
        // for each Y[i] corresponds to all X[j]
        for(size_t j = inz_begin; j < inz_end; ++j) {
            size_t c = X_indices_m[j]; // c := mode-n index ??? [1, size(mode-n)]
            // We will cut a chunk on Y into several subchunks,
            // a subchunk in Y corresponds to a chunk in X
            for(size_t r = 0; r < Y_num_subchunks; ++r) {
                // Iterate elements from each subchunk in Y
                for(size_t k = 0; k < Y_subchunk_size; ++k) {
                    /*
                    Y.offset_to_indices(idxY.get(), i * Y.chunk_size + r * Y_subchunk_size + k);
                    X.offset_to_indices(idxX.get(), j * X.chunk_size + k);
                    U.offset_to_indices(idxU.get(), r * Ustride + c);
                    std::fprintf(stderr, "Y[%s] += X[%s] * U[%s]\n", array_to_string(idxY.get(), nmodes).c_str(), array_to_string(idxX.get(), nmodes).c_str(), array_to_string(idxU.get(), nmodes).c_str());
                    */
                    if(r < nrows && c < ncols) {
                        Y_values[i * Y.chunk_size + r * Y_subchunk_size + k] += X_values[j * X.chunk_size + k] * U_values[r * Ustride + c];
                    }
                }
            }
        }
    }

    timer_kernel.stop();
    timer_kernel.print_elapsed_time("CPU TTM Kernel");

    return Y;
}

}
