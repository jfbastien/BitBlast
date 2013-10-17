typedef int i32;
typedef unsigned u32;
typedef unsigned char u8;

#define I __attribute__((always_inline))
#define assert(P) do if (!(P)) __builtin_unreachable(); while (0)

I i32 min_1(i32 x, i32 y) { return (x < y) ? x : y; }
I i32 min_2(i32 x, i32 y) { return y ^ ((x ^ y) & -(x < y)); }
bool CHECK_min_1_2(i32 x, i32 y) { return min_1(x, y) == min_2(x, y); }

I i32 max_1(i32 x, i32 y) { return (x > y) ? x : y; }
I i32 max_2(i32 x, i32 y) { return x ^ ((x ^ y) & -(x < y)); }
bool CHECK_max_1_2(i32 x, i32 y) { return max_1(x, y) == max_2(x, y); }

I u32 set_or_clear_1(bool f, u32 m, u32 w) { return f ? w | m : w & ~m; }
I u32 set_or_clear_2(bool f, u32 m, u32 w) { return w ^ ((-f ^ w) & m); }
I u32 set_or_clear_3(bool f, u32 m, u32 w) { return (w & ~m) | (-f & m); }
bool CHECK_set_or_clear_1_2(bool f, u32 m, u32 w) {
  return set_or_clear_1(f, m, w) == set_or_clear_2(f, m, w);
}
bool CHECK_set_or_clear_1_3(bool f, u32 m, u32 w) {
  return set_or_clear_1(f, m, w) == set_or_clear_3(f, m, w);
}
bool CHECK_set_or_clear_2_3(bool f, u32 m, u32 w) {
  return set_or_clear_2(f, m, w) == set_or_clear_3(f, m, w);
}

I u32 count_bits_set_1(u32 v) {
  u32 c = 0;
  for (u32 i = 0; i != sizeof(v) * 8; ++i)
    if (v & (1u << i))
      ++c;
  return c;
}
/*u32 count_bits_set_2(u32 v) {
  assert((v & 0x3ffff) == v); // Max 14 bits.
  return (v * 0x200040008001ULL & 0x111111111111111ULL) % 0xf;
  }*/
/*u32 count_bits_set_3(u32 v) {
  u32 c;
  assert((v & 0xffffff) == v); // Max 24 bits.
  c =  ((v & 0xfff) * 0x1001001001001ULL & 0x84210842108421ULL) % 0x1f;
  return c + (((v & 0xfff000) >> 12) * 0x1001001001001ULL & 0x84210842108421ULL) % 0x1f;
  }*/
/*u32 count_bits_set_4(u32 v) {
  u32 c;
  c =  ((v & 0xfff) * 0x1001001001001ULL & 0x84210842108421ULL) % 0x1f;
  c += (((v & 0xfff000) >> 12) * 0x1001001001001ULL & 0x84210842108421ULL) % 0x1f;
  return c + ((v >> 24) * 0x1001001001001ULL & 0x84210842108421ULL) % 0x1f;
  }*/
I u32 count_bits_set_5(u32 v) {
  u32 c;
  c = v - ((v >> 1) & 0x55555555);
  c = ((c >> 2) & 0x33333333) + (c & 0x33333333);
  c = ((c >> 4) + c) & 0x0F0F0F0F;
  c = ((c >> 8) + c) & 0x00FF00FF;
  c = ((c >> 16) + c) & 0x0000FFFF;
  return c;
}
bool CHECK_count_bits_set_1_5(u32 u) {
  return count_bits_set_1(u) == count_bits_set_5(u);
}

I bool parity_1(u32 v) {
  u32 c = 0;
  for (u32 i = 0; i != sizeof(v) * 8; ++i)
    if (v & (1u << i))
      ++c;
  return c & 1;
}
/*bool parity_2(u32 v) {
  assert((v & 0xff) == v); // Max 8 bits.
  return (((v * 0x0101010101010101ULL) & 0x8040201008040201ULL) % 0x1FF) & 1;
  }*/
I bool parity_3(u32 v) {
  v ^= v >> 1; v ^= v >> 2; v = (v & 0x11111111U) * 0x11111111U;
  return (v >> 28) & 1;
}
I bool parity_4(u32 v) {
  v ^= v >> 16; v ^= v >> 8; v ^= v >> 4; v &= 0xf;
  return (0x6996 >> v) & 1;
}
bool CHECK_parity_1_3(u32 v) { return parity_1(v) == parity_3(v); }
bool CHECK_parity_1_4(u32 v) { return parity_1(v) == parity_4(v); }
bool CHECK_parity_3_4(u32 v) { return parity_3(v) == parity_4(v); }

struct P {
  i32 a, b;
  bool operator==(const P&r) const { return r.a == a && r.b == b; }
};
I P swap_1(P v) { i32 temp = v.a; v.a = v.b; v.b = temp; return v; }
I P swap_2(P v) { v.a -= v.b; v.b += v.a; v.a = v.b - v.a; return v; }
I P swap_3(P v) { v.a ^= v.b; v.b ^= v.a; v.a ^= v.b; return v; }
bool CHECK_swap_1_2(P v) { return swap_1(v) == swap_2(v); }
bool CHECK_swap_1_3(P v) { return swap_1(v) == swap_3(v); }
bool CHECK_swap_2_3(P v) { return swap_2(v) == swap_3(v); }

I u8 reverse_1(u8 v) {
  u8 r = 0;
  for (u32 i = 0; i != sizeof(v) * 8; ++i)
    r |= (v >> i) << (sizeof(v) * 8 - i - 1);
  return r;
}
//u8 reverse_2(u8 v) { return (v * 0x0202020202ULL & 0x010884422010ULL) % 1023; }
I u8 reverse_3(u8 v) { return ((v * 0x80200802ULL) & 0x0884422110ULL) * 0x0101010101ULL >> 32; }
I u8 reverse_4(u8 v) { return ((v * 0x0802LU & 0x22110LU) | (v * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16; }
bool CHECK_reverse_1_3(u8 v) { return reverse_1(v) == reverse_3(v); }
bool CHECK_reverse_1_4(u8 v) { return reverse_1(v) == reverse_4(v); }
bool CHECK_reverse_3_4(u8 v) { return reverse_3(v) == reverse_4(v); }
