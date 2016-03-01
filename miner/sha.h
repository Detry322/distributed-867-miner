#ifndef _sha_
#define _sha_

#include "./structs.h"
#include <stdint.h>

void prepare_base(sha_base* base, uint8_t parentid[32], uint8_t root[32], uint64_t difficulty, uint64_t timestamp, uint8_t version);

uint64_t sha256(uint8_t parentid[32], uint8_t root[32], uint64_t difficulty, uint64_t timestamp, uint64_t nonce, uint8_t version);

#endif  // _sha_
