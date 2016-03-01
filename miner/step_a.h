#ifndef _step_a_
#define _step_a_

#include <stdint.h>
#include "./structs.h"

extern __device__ two_way_collision solution;

__global__ void step_a_kernel(sha_base* input, two_way_collision *solution, uint64_t base_nonce);

#endif // _step_a_
