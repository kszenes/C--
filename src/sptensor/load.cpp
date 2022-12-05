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
#include <cstdio>
#include <memory>
#include <ParTI/error.hpp>

#include <unistd.h>

namespace pti {

SparseTensor SparseTensor::load(std::FILE* fp, size_t start_index, size_t n_lines) {
    int io_result;

    size_t nmodes;
    io_result = std::fscanf(fp, "%zu", &nmodes);
    ptiCheckOSError(io_result != 1);

    std::unique_ptr<size_t[]> coordinate(new size_t [nmodes]);
    for(size_t m = 0; m < nmodes; ++m) {
        io_result = std::fscanf(fp, "%zu", &coordinate[m]);
        ptiCheckOSError(io_result != 1);
    }

    std::unique_ptr<bool[]> mode_is_dense(new bool [nmodes]());

    SparseTensor tensor(nmodes, coordinate.get(), mode_is_dense.get());
    // preallocate space
    if (n_lines) {
        tensor.reserve(n_lines);
        tensor.values_thrust_h = thrust::host_vector<Scalar>(n_lines);
        tensor.values_thrust_d = thrust::device_vector<Scalar>(n_lines);

        tensor.modes_h = {thrust::host_vector<IndexType>(n_lines),
                          thrust::host_vector<IndexType>(n_lines), 
                          thrust::host_vector<IndexType>(n_lines)};
        tensor.modes_d = {thrust::device_vector<IndexType>(n_lines),
                          thrust::device_vector<IndexType>(n_lines),
                          thrust::device_vector<IndexType>(n_lines)};

    }

    int i = 0;
    for(;;) {
        for(size_t m = 0; m < nmodes; ++m) {
            io_result = std::fscanf(fp, "%zu", &coordinate[m]);
            if(io_result != 1) break;
            coordinate[m] -= start_index;
        }
        double value;
        io_result = std::fscanf(fp, "%lg", &value);
        if(io_result != 1) break;
        // tensor.append(coordinate.get(), value);
        // `put` if preallocated
        tensor.put(i, coordinate.get(), value);
        ++i;

    }
    ptiCheckOSError(io_result != 1 && !std::feof(fp));

    for (int i = 0; i < tensor.modes_h.size(); ++i) {
        tensor.modes_d[i] = tensor.modes_h[i];
    }
    tensor.values_thrust_d = tensor.values_thrust_h;

    tensor.zip_it_d = thrust::make_zip_iterator(thrust::make_tuple(
        tensor.modes_d[0].begin(), tensor.modes_d[1].begin(), tensor.modes_d[2].begin()
    ));
    tensor.sort_thrust(true);

    // tensor.sort_index();

    return tensor;
}

}
