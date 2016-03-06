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
  uint64_t chain2 = triples[thread_id].chains[1].start;
  int32_t chain2_length = triples[thread_id].chains[1].length;
  uint64_t chain3 = triples[thread_id].chains[2].start;
  int32_t chain3_length = triples[thread_id].chains[2].length;
  printf("= %d %d %d", chain1_length, chain2_length, chain3_length);
  for (int i = 0; i < chain1_length - chain2_length; i++) {
    chain1 = calculate_sha_b(chain1);
  }
  chain1_length = chain2_length;
  for (int i = 0; i < chain2_length - chain3_length; i++) {
    chain1 = calculate_sha_b(chain1);
    chain2 = calculate_sha_b(chain2);
  }
  if (chain1 == chain2)
    return;
  printf("= made it this far...\n");
  uint64_t t1, t2, t3;
  for (int i = 0; i < chain3_length; i++) {
    t1 = calculate_sha_b(chain1);
    t2 = calculate_sha_b(chain2);
    t3 = calculate_sha_b(chain3);
    if (t1 == t2 && t2 == t3) {
      printf("B %lu %lu %lu %lu\n", bbase.timestamp, chain1, chain2, chain3);
      return;
    } else if (t1 == t2) {
      printf("= exit early...\n");
      return;
    } else if (t2 == t3) {
      printf("= exit early...\n");
      return;
    } else if (t1 == t3) {
      printf("= exit early...\n");
      return;
    }
    chain1 = t1;
    chain2 = t2;
    chain3 = t3;
  }
  printf("= What the dicks\n");
};
