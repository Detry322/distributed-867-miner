
#include "./sha_calculate.h"
#include "./sha.h"
#include <cstdio>

void prepare_base(sha_base* base, uint8_t parentid[32], uint8_t root[32], uint64_t difficulty, uint64_t timestamp, uint8_t version) {

  uint32_t work[64];
  uint32_t A, B, C, D, E, F, G, H, t1, t2;
  uint32_t h0, h1, h2, h3, h4, h5, h6, h7;

  h0 = A = 0x6a09e667;
  h1 = B = 0xbb67ae85;
  h2 = C = 0x3c6ef372;
  h3 = D = 0xa54ff53a;
  h4 = E = 0x510e527f;
  h5 = F = 0x9b05688c;
  h6 = G = 0x1f83d9ab;
  h7 = H = 0x5be0cd19;

  int i;
  for (i = 0; i < 8; i++) {
    work[i] = prepare_big_endian(parentid, 4*i);
  }

  for (i = 0; i < 8; i++) {
    work[i+8] = prepare_big_endian(root, 4*i);
  }

  SHA_CALCULATE(work, A, B, C, D, E, F, G, H);

  base->h[0] = h0 + A;
  base->h[1] = h1 + B;
  base->h[2] = h2 + C;
  base->h[3] = h3 + D;
  base->h[4] = h4 + E;
  base->h[5] = h5 + F;
  base->h[6] = h6 + G;
  base->h[7] = h7 + H;

  insert_long_big_endian(difficulty, base->work, 0);
  base->difficulty = difficulty;
  insert_long_big_endian(timestamp, base->work, 2);
  // in->work[4] and [5] are for the nonce
  base->work[6] = (version << 24) | 0x800000;
  base->work[7] = 0;
  base->work[8] = 0;
  base->work[9] = 0;
  base->work[10] = 0;
  base->work[11] = 0;
  base->work[12] = 0;
  base->work[13] = 0;
  insert_long_big_endian(712L, base->work, 14);
}

uint64_t sha256(uint8_t parentid[32], uint8_t root[32], uint64_t difficulty, uint64_t timestamp, uint64_t nonce, uint8_t version) {

  sha_base base;
  prepare_base(&base, parentid, root, difficulty, timestamp, version);
  uint32_t work[64];
  uint32_t A, B, C, D, E, F, G, H, t1, t2;
  A = base.h[0];
  B = base.h[1];
  C = base.h[2];
  D = base.h[3];
  E = base.h[4];
  F = base.h[5];
  G = base.h[6];
  H = base.h[7];

  int i;
  for (i = 0; i < 16; i++) {
    work[i] = base.work[i];
  }

  insert_long_big_endian(nonce, work, 4);

  SHA_CALCULATE(work, A, B, C, D, E, F, G, H);

  uint64_t final_hash = ((((uint64_t) (base.h[6] + G)) << 32) | ((uint64_t) (base.h[7] + H))) & ((1L << difficulty) - 1);
  return final_hash;
}
