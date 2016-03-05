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
  uint64_t chain1 = triples[thread_id].chains[0].start;
  int32_t chain1_length = triples[thread_id].chains[0].length;
  uint64_t chain2 = triples[thread_id].chains[0].start;
  int32_t chain2_length = triples[thread_id].chains[0].length;
  uint64_t chain3 = triples[thread_id].chains[0].start;
  int32_t chain3_length = triples[thread_id].chains[0].length;
  while (true) {
    if (chain1_length > chain2_length) {
      chain1 = calculate_sha_b(chain1);
      chain1_length--;
      continue;
    }
    if (chain1 == chain2) {
      printf("= Not found early \n");
      return;
    }
    // If it gets here, chain1.length == chain2.length
    if (chain1_length > chain3_length) {
      chain1 = calculate_sha_b(chain1);
      chain1_length--;
      chain2 = calculate_sha_b(chain2);
      chain2_length--;
      continue;
    }
    break;
  };
  if (chain1 == chain2 || chain1 == chain3 || chain2 == chain3) {
    printf("= Not found before second loop\n");
    return;
  }
  // checking for collisions;
  uint64_t t1, t2, t3;
  for (int i = 0; i < chain1_length; i++) {
    t1 = calculate_sha_b(chain1);
    t2 = calculate_sha_b(chain2);
    t3 = calculate_sha_b(chain3);
    if (t1 == t2 && t2 == t3) {
      printf("B %lu %lu %lu %lu\n", bbase.timestamp, chain1, chain2, chain3);
      return;
    } else if (t1 == t2) {
      printf("= Not found, 1 and 2 collided\n");
      return;
    } else if (t2 == t3) {
      printf("= Not found, 2 and 3 collided\n");
      return;
    } else if (t1 == t3) {
      printf("= Not found, 1 and 3 collided\n");
      return;
    }
    chain1 = t1;
    chain2 = t2;
    chain3 = t3;
  }
};
