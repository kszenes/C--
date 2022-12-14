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
#include <ParTI/sptensor.hpp>
#include <ParTI/timer.hpp>
using namespace pti;

int main(int argc, char const* argv[]) {
	bool dense_format = false;
	size_t limit = 10;
	Scalar times = 2;
	ParamDefinition defs[] = {
		{ "-d", PARAM_BOOL, { &dense_format } },
		{ "--dense-format", PARAM_BOOL, { &dense_format } },
		{ "-l", PARAM_SIZET, { &limit } },
		{ "--limit", PARAM_SIZET, { &limit } },
		{ "-s", PARAM_SCALAR, { &times} }, 
		{ ptiEndParamDefinition }
	};
	std::vector<char const*> args = parse_args(argc, argv, defs);

	if (args.size() != 1 && args.size() != 2) {
		std::printf("Usage: %s [OPTIONS] input_tensor [output_tensor]\n\n", argv[0]);
		std::printf("Options:\n");
		std::printf("\t-d, --dense-format\tPrint tensor in dense format instead of sparse format.\n");
		std::printf("\t-l, --limit\t\tLimit the number of elements to print [Default: 10].\n");
		std::printf("\t-s, [Default: 2].\n");
		std::printf("\n");
		return 1;
	}


	CFile fM(args[0], "r");
	SparseTensor tmc = SparseTensor::load(fM, 1);
	fM.fclose();

	std::printf("tmc = %s\n", tmc.to_string(!dense_format, limit).c_str());
	
	Timer timer(cpu);
	timer.start();
	SparseTensor Y = tensor_multiply_scalar(tmc, times);
	timer.stop();
	std::printf("Y = %s\n", Y.to_string(!dense_format, limit).c_str());
	timer.print_elapsed_time("tensor multiply scalar time");
	if (args.size() == 2) {
		CFile fY(args[1], "w");
		tmc.dump(fY, 1);
	}
	return 0;
}
