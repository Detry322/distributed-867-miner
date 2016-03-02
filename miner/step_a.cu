#include "./step_a.h"
#include "./sha_calculate.h"
#include <stdint.h>
#include <assert.h>
#include <stdio.h>

__shared__ sha_base abase;

__device__ uint64_t calculate_sha_a(uint64_t nonce) {
  uint32_t work[64];
  uint32_t A, B, C, D, E, F, G, H, t1, t2;
  #pragma unroll
  for (int i = 0; i < 16; i++)
    work[i] = abase.work[i];
  A = abase.h[0];
  B = abase.h[1];
  C = abase.h[2];
  D = abase.h[3];
  E = abase.h[4];
  F = abase.h[5];
  G = abase.h[6];
  H = abase.h[7];
  insert_long_big_endian(nonce, work, 4);
  SHA_CALCULATE(work, A, B, C, D, E, F, G, H);
  return ((((uint64_t) (abase.h[6] + G)) << 32) | ((uint64_t) (abase.h[7] + H))) & ((1L << abase.difficulty) - 1);
}

__global__ void step_a_kernel(sha_base* input, uint64_t base_nonce) {
  if (threadIdx.x == 0) {
    abase = *input;
  }
  __syncthreads();
  uint64_t initial_nonce = base_nonce + (blockIdx.x*blockDim.x+threadIdx.x);
  uint64_t distinguished_cutoff = 1L << (abase.difficulty*2/3);
  int64_t max_steps = 2L << (abase.difficulty/3);
  uint64_t d = initial_nonce;
  for (int64_t i = 1; i < max_steps; i++) {
    d = calculate_sha_a(d);
    if (d < distinguished_cutoff) {
      printf("A %lu %lu %lu %ld\n", abase.timestamp, initial_nonce, d, i);
      return;
    }
  }
};
