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

#include <ParTI/sptensor.hpp>
#include <cstring>
#include <memory>
#include <utility>

namespace pti {

namespace {

int compare_indices(SparseTensor& tsr, IndexType i, IndexType j) {
    for(IndexType m = 0; m < tsr.sparse_order.size(); ++m) {
        IndexType mode = tsr.sparse_order(cpu)[m];
        IndexType idx_i = tsr.indices[mode](cpu)[i];
        IndexType idx_j = tsr.indices[mode](cpu)[j];
        if(idx_i < idx_j) {
            return -1;
        } else if(idx_i > idx_j) {
            return 1;
        }
    }
    return 0;
}

void swap_values(SparseTensor& tsr, IndexType i, IndexType j, Scalar* swap_buffer) {
    for(IndexType m = 0; m < tsr.nmodes; ++m) {
        if(!tsr.is_dense(cpu)[m]) {
            std::swap(tsr.indices[m](cpu)[i], tsr.indices[m](cpu)[j]);
        }
    }
    Scalar* value_i = &tsr.values(cpu)[i * tsr.chunk_size];
    Scalar* value_j = &tsr.values(cpu)[j * tsr.chunk_size];
    std::memcpy(swap_buffer, value_i,     tsr.chunk_size * sizeof (Scalar));
    std::memcpy(value_i,     value_j,     tsr.chunk_size * sizeof (Scalar));
    std::memcpy(value_j,     swap_buffer, tsr.chunk_size * sizeof (Scalar));
}

void quick_sort_index(SparseTensor& tsr, IndexType l, IndexType r, Scalar* swap_buffer) {
    IndexType i, j, p;
    if(r-l < 2) {
        return;
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        while(compare_indices(tsr, i, p) < 0) {
            ++i;
        }
        while(compare_indices(tsr, p, j) < 0) {
            --j;
        }
        if(i >= j) {
            break;
        }
        swap_values(tsr, i, j, swap_buffer);
        if(i == p) {
            p = j;
        } else if(j == p) {
            p = i;
        }
    }
    quick_sort_index(tsr, l, i, swap_buffer);
    quick_sort_index(tsr, i, r, swap_buffer);
}

}

void SparseTensor::sort_index() {
    std::unique_ptr<Scalar[]> swap_buffer(new Scalar [chunk_size]);

    quick_sort_index(*this, 0, num_chunks, swap_buffer.get());
}

void SparseTensor::sort_index(IndexType const sparse_order[]) {
    std::memcpy(this->sparse_order(cpu), sparse_order, this->sparse_order.size() * sizeof (IndexType));

    std::unique_ptr<Scalar[]> swap_buffer(new Scalar [chunk_size]);

    quick_sort_index(*this, 0, num_chunks, swap_buffer.get());
}

// struct sort_indices_thrust {
//     // 3d
//     __device__ __host__ bool operator()(const ulong3 &a, const ulong3 &b){
//         return (a.y < b.y) || (a.y == b.y && a.z < b.z);
//     }
//     // 4d
//     // __device__ __host__ bool operator()(const Color &a, const Color &b){
//     //     return (a.y < b.y) || (a.y == b.y && a.z < b.z) || (a.y == b.y && a.z == b.z && a.w < b.w);
//     // }
//     // 2d (a.y < b.y)
//     // 3d (a.y < b.y) || (a.y == b.y && a.z < b.z)
// };

void SparseTensor::sort_thrust(bool cuda_dev) {
    auto sort_zip_mode0 = []__host__ __device__ (
        const IndexTuple& a, const IndexTuple& b 
    ) {
        return (thrust::get<1>(a) < thrust::get<1>(b)) ||
               (thrust::get<1>(a) == thrust::get<1>(b) && thrust::get<2>(a) < thrust::get<2>(b));
    };

    if (cuda_dev) {
        thrust::sort_by_key(
            zip_it_d, zip_it_d + chunk_size * num_chunks,
            values_thrust_d.begin(),
            sort_zip_mode0
        );
    }
    // else {
    //     thrust::sort_by_key(
    //         indices_thrust_h.begin(),
    //         indices_thrust_h.end(),
    //         values_thrust_h.begin(),
    //         pti::sort_indices_thrust()
    //     );
    // }

}


}
