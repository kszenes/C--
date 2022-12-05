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

#ifndef SPTENSOR_H
#define SPTENSOR_H

#include <cstddef>
#include <cstdio>
#include <string>
#include <ParTI/base_tensor.hpp>
#include <ParTI/scalar.hpp>
#include <ParTI/memblock.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>

namespace pti {

struct Tensor;

struct SparseTensor : public BaseTensor {

    MemBlock<bool[]> is_dense;

    MemBlock<size_t[]> dense_order;

    MemBlock<size_t[]> sparse_order;

    MemBlock<size_t[]> strides;

    size_t chunk_size; // product of strides

    size_t num_chunks;

    MemBlock<size_t[]>* indices;
    
    std::vector<thrust::host_vector<IndexType>> modes_h;
    std::vector<thrust::device_vector<IndexType>> modes_d;

    using IntIterator = thrust::device_vector<IndexType>::iterator;
    using IteratorTuple = thrust::tuple<IntIterator, IntIterator, IntIterator>;
    using IndexTuple = thrust::tuple<IndexType, IndexType, IndexType>;
    using ZipIterator = thrust::zip_iterator<IteratorTuple>;

    using IteratorTupleMode = thrust::tuple<IntIterator, IntIterator>;
    using ZipIteratorMode = thrust::zip_iterator<IteratorTuple>;

    ZipIterator zip_it_d;

    MemBlock<Scalar[]> values;
    thrust::host_vector<Scalar> values_thrust_h;
    thrust::device_vector<Scalar> values_thrust_d;

public:

    explicit SparseTensor();
    explicit SparseTensor(size_t nmodes, size_t const shape[], bool const is_dense[]);
    SparseTensor(SparseTensor&& other);
    SparseTensor& operator= (SparseTensor&& other);
    ~SparseTensor();

    explicit SparseTensor(Tensor&& other);
    SparseTensor& operator= (Tensor&& other);

    SparseTensor clone();
    SparseTensor& reset(size_t nmodes, size_t const shape[], bool const is_dense[]);

    bool offset_to_indices(size_t indices[], size_t offset);
    size_t indices_to_intra_offset(size_t const indices[]);

    void dump(std::FILE* fp, size_t start_index = 0);

    static SparseTensor load(std::FILE* fp, size_t start_index = 0, size_t n_lines = 0);

    std::string to_string(bool sparse_format, size_t limit = 0);

    void append(size_t const coord[], Scalar const value[]);
    void append(size_t const coord[], Scalar value);
    void put(size_t const location, size_t const coord[], Scalar const value[]);
    void put(size_t const location, size_t const coord[], Scalar value);
    size_t reserve(size_t size, bool initialize = true);
    void init_single_chunk(bool initialize = true);

    void sort_index();
    void sort_index(size_t const sparse_order[]);

    void sort_thrust(bool cuda_dev = false);

    SparseTensor to_fully_sparse();
    SparseTensor to_fully_dense();

    double norm(Device *device);

};

}

#endif /* SPTENSOR_H */
