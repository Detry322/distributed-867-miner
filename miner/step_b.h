#ifndef _step_b_
#define _step_b_

#include <stdint.h>
#include "./structs.h"

__global__ void step_b_kernel(sha_base* input, triple* triples);

#endif // _step_b_
