#ifndef _sha_calculate_
#define _sha_calculate_

#define byteswap(x) (((x>>24) & 0x000000ff) | ((x>>8) & 0x0000ff00) | ((x<<8) & 0x00ff0000) | ((x<<24) & 0xff000000))
#define prepare_big_endian(bytearr, i) ((((bytearr)[i]) << 24) | (((bytearr)[i+1]) << 16) | (((bytearr)[i+2]) << 8) | (((bytearr)[i+3])))
#define reverse_word(word) ((((word) & 0xff000000) >> 24) | (((word) & 0x00ff0000) >> 8) | (((word) & 0x0000ff00) << 8) | (((word) & 0x000000ff) << 24))
#define insert_long_big_endian(l, work, i) { (work)[i] = (unsigned int) ((l) >> 32); (work)[i+1] = (unsigned int) ((l) & 0x00000000FFFFFFFFL); }
#define rotateright(x,bits) (((x) >> bits) + (x << (32 - bits)))
#define R(x) (work[x] = (rotateright((work[x-2]^rotateright(work[x-2],2)),17)^((work[x-2])>>10)) + work[x -  7] + (rotateright((work[x-15]^rotateright(work[x-15],11)),7)^((work[x-15])>>3)) + work[x - 16])
#define sharound(a,b,c,d,e,f,g,h,x,K) {t1=h+rotateright((e^rotateright((e^rotateright(e,14)),5)),6)+(g^(e&(f^g))) /*((g&(~e))|(e&f))*/ +K+x; t2=rotateright((a^rotateright((a^rotateright(a,9)),11)),2)+/*((a&b)|(c&(a|b)))*/ (a^((a^b)&(a^c))); d+=t1; h=t1+t2;}

#define SHA_CALCULATE(work, A, B, C, D, E, F, G, H) \
do { \
  sharound(A,B,C,D,E,F,G,H,work[0],0x428a2f98); \
  sharound(H,A,B,C,D,E,F,G,work[1],0x71374491); \
  sharound(G,H,A,B,C,D,E,F,work[2],0xb5c0fbcf); \
  sharound(F,G,H,A,B,C,D,E,work[3],0xe9b5dba5); \
  sharound(E,F,G,H,A,B,C,D,work[4],0x3956c25b); \
  sharound(D,E,F,G,H,A,B,C,work[5],0x59f111f1); \
  sharound(C,D,E,F,G,H,A,B,work[6],0x923f82a4); \
  sharound(B,C,D,E,F,G,H,A,work[7],0xab1c5ed5); \
  sharound(A,B,C,D,E,F,G,H,work[8],0xd807aa98); \
  sharound(H,A,B,C,D,E,F,G,work[9],0x12835b01); \
  sharound(G,H,A,B,C,D,E,F,work[10],0x243185be); \
  sharound(F,G,H,A,B,C,D,E,work[11],0x550c7dc3); \
  sharound(E,F,G,H,A,B,C,D,work[12],0x72be5d74); \
  sharound(D,E,F,G,H,A,B,C,work[13],0x80deb1fe); \
  sharound(C,D,E,F,G,H,A,B,work[14],0x9bdc06a7); \
  sharound(B,C,D,E,F,G,H,A,work[15],0xc19bf174); \
  R(16); \
  R(17); \
  R(18); \
  R(19); \
  R(20); \
  R(21); \
  R(22); \
  R(23); \
  sharound(A,B,C,D,E,F,G,H,work[16],0xe49b69c1); \
  sharound(H,A,B,C,D,E,F,G,work[17],0xefbe4786); \
  sharound(G,H,A,B,C,D,E,F,work[18],0x0fc19dc6); \
  sharound(F,G,H,A,B,C,D,E,work[19],0x240ca1cc); \
  sharound(E,F,G,H,A,B,C,D,work[20],0x2de92c6f); \
  sharound(D,E,F,G,H,A,B,C,work[21],0x4a7484aa); \
  sharound(C,D,E,F,G,H,A,B,work[22],0x5cb0a9dc); \
  sharound(B,C,D,E,F,G,H,A,work[23],0x76f988da); \
  R(24); \
  R(25); \
  R(26); \
  R(27); \
  R(28); \
  R(29); \
  R(30); \
  R(31); \
  sharound(A,B,C,D,E,F,G,H,work[24],0x983e5152); \
  sharound(H,A,B,C,D,E,F,G,work[25],0xa831c66d); \
  sharound(G,H,A,B,C,D,E,F,work[26],0xb00327c8); \
  sharound(F,G,H,A,B,C,D,E,work[27],0xbf597fc7); \
  sharound(E,F,G,H,A,B,C,D,work[28],0xc6e00bf3); \
  sharound(D,E,F,G,H,A,B,C,work[29],0xd5a79147); \
  sharound(C,D,E,F,G,H,A,B,work[30],0x06ca6351); \
  sharound(B,C,D,E,F,G,H,A,work[31],0x14292967); \
  R(32); \
  R(33); \
  R(34); \
  R(35); \
  R(36); \
  R(37); \
  R(38); \
  R(39); \
  sharound(A,B,C,D,E,F,G,H,work[32],0x27b70a85); \
  sharound(H,A,B,C,D,E,F,G,work[33],0x2e1b2138); \
  sharound(G,H,A,B,C,D,E,F,work[34],0x4d2c6dfc); \
  sharound(F,G,H,A,B,C,D,E,work[35],0x53380d13); \
  sharound(E,F,G,H,A,B,C,D,work[36],0x650a7354); \
  sharound(D,E,F,G,H,A,B,C,work[37],0x766a0abb); \
  sharound(C,D,E,F,G,H,A,B,work[38],0x81c2c92e); \
  sharound(B,C,D,E,F,G,H,A,work[39],0x92722c85); \
  R(40); \
  R(41); \
  R(42); \
  R(43); \
  R(44); \
  R(45); \
  R(46); \
  R(47); \
  sharound(A,B,C,D,E,F,G,H,work[40],0xa2bfe8a1); \
  sharound(H,A,B,C,D,E,F,G,work[41],0xa81a664b); \
  sharound(G,H,A,B,C,D,E,F,work[42],0xc24b8b70); \
  sharound(F,G,H,A,B,C,D,E,work[43],0xc76c51a3); \
  sharound(E,F,G,H,A,B,C,D,work[44],0xd192e819); \
  sharound(D,E,F,G,H,A,B,C,work[45],0xd6990624); \
  sharound(C,D,E,F,G,H,A,B,work[46],0xf40e3585); \
  sharound(B,C,D,E,F,G,H,A,work[47],0x106aa070); \
  R(48); \
  R(49); \
  R(50); \
  R(51); \
  R(52); \
  R(53); \
  R(54); \
  R(55); \
  sharound(A,B,C,D,E,F,G,H,work[48],0x19a4c116); \
  sharound(H,A,B,C,D,E,F,G,work[49],0x1e376c08); \
  sharound(G,H,A,B,C,D,E,F,work[50],0x2748774c); \
  sharound(F,G,H,A,B,C,D,E,work[51],0x34b0bcb5); \
  sharound(E,F,G,H,A,B,C,D,work[52],0x391c0cb3); \
  sharound(D,E,F,G,H,A,B,C,work[53],0x4ed8aa4a); \
  sharound(C,D,E,F,G,H,A,B,work[54],0x5b9cca4f); \
  sharound(B,C,D,E,F,G,H,A,work[55],0x682e6ff3); \
  R(56); \
  R(57); \
  R(58); \
  R(59); \
  R(60); \
  R(61); \
  R(62); \
  R(63); \
  sharound(A,B,C,D,E,F,G,H,work[56],0x748f82ee); \
  sharound(H,A,B,C,D,E,F,G,work[57],0x78a5636f); \
  sharound(G,H,A,B,C,D,E,F,work[58],0x84c87814); \
  sharound(F,G,H,A,B,C,D,E,work[59],0x8cc70208); \
  sharound(E,F,G,H,A,B,C,D,work[60],0x90befffa); \
  sharound(D,E,F,G,H,A,B,C,work[61],0xa4506ceb); \
  sharound(C,D,E,F,G,H,A,B,work[62],0xbef9a3f7); \
  sharound(B,C,D,E,F,G,H,A,work[63],0xc67178f2); \
} while(0);

#endif  // _sha_calculate_
