#ifndef _step_a_
#define _step_a_

#include <stdint.h>
#include "./structs.h"

__global__ void step_a_kernel(sha_base* input, uint64_t base_nonce);

#endif // _step_a_
