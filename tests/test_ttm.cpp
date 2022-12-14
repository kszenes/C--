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
#include <ParTI/argparse.hpp>
#include <ParTI/cfile.hpp>
#include <ParTI/timer.hpp>
#include <ParTI/session.hpp>
#include <ParTI/sptensor.hpp>
#include <ParTI/tensor.hpp>

using namespace pti;


int main(int argc, char const* argv[]) {
    size_t mode = 0;
    bool dense_format = false;
    size_t limit = 10;
    int device = 0;
    ParamDefinition defs[] = {
        { "-m",             PARAM_SIZET, { &mode } },
        { "--mode",         PARAM_SIZET, { &mode } },
        { "-d",             PARAM_BOOL,  { &dense_format } },
        { "--dense-format", PARAM_BOOL,  { &dense_format } },
        { "-l",             PARAM_SIZET, { &limit } },
        { "--limit",        PARAM_SIZET, { &limit } },
        { "--dev",          PARAM_INT,    { &device } },
        { ptiEndParamDefinition }
    };
    std::vector<char const*> args = parse_args(argc, argv, defs);

    if(args.size() != 2 && args.size() != 3) {
        std::printf("Usage: %s [OPTIONS] X U [Y]\n\n", argv[0]);
        std::printf("Options:\n");
        std::printf("\t-m, --mode\tUse specific mode for multiplication [Default: 0]\n");
        std::printf("\t-d, --dense-format\tPrint tensor in dense format instead of sparse format.\n");
        std::printf("\t-l, --limit\t\tLimit the number of elements to print [Default: 10].\n");
        std::printf("\t--dev\t\tComputing device\n");
        std::printf("\n");
        return 1;
    }

    session.print_devices();

    CFile fX(args[0], "r");
    CFile fLines(args[0], "r");
    int number_of_lines = 0;
    int ch;
    while (EOF != (ch=getc(fLines)))
        if ('\n' == ch)
            ++number_of_lines;
    printf("nnz(X) = %u\n", number_of_lines - 2);
    SparseTensor X = SparseTensor::load(fX, 1, number_of_lines - 2);
    fLines.fclose();
    fX.fclose();
    // for (int i = 0; i < number_of_lines - 2; ++i) {
    //     std::cout << X.modes_d[0][i] << ' ' << X.modes_d[1][i] << ' ' << X.modes_d[2][i] << ' ' << X.values_thrust_d[i] << '\n';
    // }
    // Timer sort_timer(cpu);
    // sort_timer.start();
    // X.sort_thrust(true);
    // sort_timer.stop();
    // sort_timer.print_elapsed_time("CUDA Sort Thrust");

    // sort_timer.start();
    // thrust::device_vector<ulong> mode_m(X.indices_thrust_d.size());
    // sort_timer.stop();
    // sort_timer.print_elapsed_time("CUDA Get mode-m Thrust");

    // exit(1);

    CFile fU(args[1], "r");
    Tensor U = Tensor::load(fU);
    fU.fclose();

    std::printf("U = %s\n", U.to_string(limit).c_str());
    transpose_matrix_inplace(U, true, false, session.devices[device]);
    std::printf("Transposing U...\n");
    std::printf("U = %s\n", U.to_string(limit).c_str());

    Timer timer(cpu);
    timer.start();
    SparseTensor Y = tensor_times_matrix(X, U, mode, session.devices[device]);
    timer.stop();

    std::printf("Y = %s\n", Y.to_string(!dense_format, limit, true).c_str());
    timer.print_elapsed_time("TTM");

    if(args.size() == 3) {
        CFile fY(args[2], "w");
        Y.dump(fY, 1);
    }

    return 0;
}
