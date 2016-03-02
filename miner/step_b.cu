#include "./step_b.h"
#include "./sha_calculate.h"
#include <stdint.h>
#include <assert.h>
#include <stdio.h>

__shared__ sha_base bbase;

__device__ uint64_t calculate_sha_b(uint64_t nonce) {
  uint32_t work[64];
  uint32_t A, B, C, D, E, F, G, H, t1, t2;
  #pragma unroll
  for (int i = 0; i < 16; i++)
    work[i] = bbase.work[i];
  A = bbase.h[0];
  B = bbase.h[1];
  C = bbase.h[2];
  D = bbase.h[3];
  E = bbase.h[4];
  F = bbase.h[5];
  G = bbase.h[6];
  H = bbase.h[7];
  insert_long_big_endian(nonce, work, 4);
  SHA_CALCULATE(work, A, B, C, D, E, F, G, H);
  return ((((uint64_t) (bbase.h[6] + G)) << 32) | ((uint64_t) (bbase.h[7] + H))) & ((1L << bbase.difficulty) - 1);
}

__global__ void step_b_kernel(sha_base* input, triple* triples) {
  if (threadIdx.x == 0) {
    bbase = *input;
  }
  __syncthreads();
  uint64_t thread_id = (blockIdx.x*blockDim.x+threadIdx.x);
  triple working = triples[thread_id];
  hash_chain chain1 = working.chains[0];
  hash_chain chain2 = working.chains[1];
  hash_chain chain3 = working.chains[2];
  // advancing chains
  while (true) {
    if (chain1.length > chain2.length) {
      chain1.start = calculate_sha_b(chain1.start);
      chain1.length--;
      continue;
    }
    if (chain1.start == chain2.start)
      return;
    // If it gets here, chain1.length == chain2.length
    if (chain1.length > chain3.length) {
      chain1.start = calculate_sha_b(chain1.start);
      chain1.length--;
      chain2.start = calculate_sha_b(chain2.start);
      chain2.length--;
      continue;
    }
    break;
  };
  if (chain1.start == chain2.start || chain1.start == chain3.start || chain2.start == chain3.start)
    return;
  // checking for collisions;
  for (int i = 0; i < chain1.length; i++) {
    uint64_t t1, t2, t3;
    t1 = calculate_sha_b(chain1.start);
    t2 = calculate_sha_b(chain2.start);
    t3 = calculate_sha_b(chain3.start);
    if (t1 == t2 && t2 == t3) {
      printf("B %lu %lu %lu %lu\n", bbase.timestamp, chain1.start, chain2.start, chain3.start);
      return;
    } else if (t1 == t2) {
      return;
    } else if (t2 == t3) {
      return;
    } else if (t1 == t3) {
      return;
    }
    chain1.start = t1;
    chain2.start = t2;
    chain3.start = t3;
  }
};
