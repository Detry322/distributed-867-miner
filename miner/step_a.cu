#include "./step_a.h"
#include "./sha_calculate.h"
#include <stdint.h>
#include <assert.h>
#include <stdio.h>

__shared__ sha_base base;

__device__ uint64_t calculate_sha(uint64_t nonce) {
  uint32_t work[64];
  uint32_t A, B, C, D, E, F, G, H, t1, t2;
  #pragma unroll
  for (int i = 0; i < 16; i++)
    work[i] = base.work[i];
  A = base.h[0];
  B = base.h[1];
  C = base.h[2];
  D = base.h[3];
  E = base.h[4];
  F = base.h[5];
  G = base.h[6];
  H = base.h[7];
  insert_long_big_endian(nonce, work, 4);
  SHA_CALCULATE(work, A, B, C, D, E, F, G, H);
  return ((((uint64_t) (base.h[6] + G)) << 32) | ((uint64_t) (base.h[7] + H))) & ((1L << base.difficulty) - 1);
}

__device__ void find_solution(uint64_t timestamp, uint64_t nonce_a, uint64_t nonce_b) {
  uint64_t temp_a, temp_b;
  while (true) {
    temp_a = calculate_sha(nonce_a);
    temp_b = calculate_sha(nonce_b);
    if (temp_a == temp_b) {
      printf("A %lu %lu %lu\n", timestamp, nonce_a, nonce_b);
      return;
    } else {
      nonce_a = temp_a;
      nonce_b = temp_b;
    }
  }
}

__global__ void step_a_kernel(sha_base* input, uint64_t base_nonce) {
  if (threadIdx.x == 0) {
    base = *input;
  }
  __syncthreads();
  uint64_t initial_nonce = base_nonce + (blockIdx.x*blockDim.x+threadIdx.x);
  uint64_t distinguished_cutoff = 1L << (base.difficulty*2/3);
  int64_t max_steps = 20L << (base.difficulty/3);
  uint64_t d = initial_nonce;
  for (int64_t i = 1; i < max_steps; i++) {
    d = calculate_sha(d);
    if (d < distinguished_cutoff) {
      printf("A %lu %lu %lu %ld\n", base.timestamp, initial_nonce, d, i);
      return;
    }
  }
};
