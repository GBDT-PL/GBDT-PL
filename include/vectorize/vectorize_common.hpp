//
//  common.h
//  LinearGBMVector
//


//

#ifndef vectorize_common_hpp
#define vectorize_common_hpp

#include "x86intrin.h"  

//extend a 8 bits unsigned char into 64 bits, replicate it 8 times
inline int64_t extend_8_to_64(uint8_t __t) {
    uint64_t __extend = __t;
    return __extend | (__extend << 8) | (__extend << 16) | (__extend << 24) |
    (__extend << 32) | (__extend << 40) | (__extend << 48) | (__extend << 56);  
}

//convert a packed 8 * 8 unsigned char vector into a packed 8 * 8 signed char vector
inline int64_t convert_8x8_to_signed(int64_t x) {
    return x ^ 0x8080808080808080;
}

inline __m256i get_maskd(int vec_size, int row_size) {
    int diff = vec_size - row_size;
    if(diff == 0) {
        return _mm256_set_epi64x(~0, ~0, ~0, ~0);
    }
    else if(diff == 1) {
        return _mm256_set_epi64x(0, ~0, ~0, ~0);
    }
    else if(diff == 2) {
        return _mm256_set_epi64x(0, 0, ~0, ~0);
    }
    else if(diff == 3) {
        return _mm256_set_epi64x(0, 0, 0, ~0);
    }
    else {
        return _mm256_set_epi64x(~0, ~0, ~0, ~0);
    }
}

inline __m128i get_masks(int vec_size, int row_size) {
    int diff = vec_size - row_size;
    if(diff == 0) {
        return _mm_set_epi32(~0, ~0, ~0, ~0);
    }
    else if(diff == 1) {
        return _mm_set_epi32(0, ~0, ~0, ~0);
    }
    else if(diff == 2) {
        return _mm_set_epi32(0, 0, ~0, ~0);
    }
    else if(diff == 3) {
        return _mm_set_epi32(0, 0, 0, ~0);
    }
    else {
        return _mm_set_epi32(~0, ~0, ~0, ~0);
    }
}

inline __m256i get_maskds(int vec_size, int row_size) {
    int diff = vec_size - row_size;
    if(diff == 0) {
        return _mm256_set_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0);
    }
    else if(diff == 1) {
        return _mm256_set_epi32(0,  ~0, ~0, ~0, ~0, ~0, ~0, ~0);
    }
    else if(diff == 2){
        return _mm256_set_epi32(0,  0, ~0, ~0, ~0, ~0, ~0, ~0);
    }
    else if(diff == 3) {
        return _mm256_set_epi32(0,  0, 0, ~0, ~0, ~0, ~0, ~0);
    }
    else if(diff == 4) {
        return _mm256_set_epi32(0, 0, 0, 0, ~0, ~0, ~0, ~0);
    }
    else if(diff == 5) {
        return _mm256_set_epi32(0, 0, 0, 0, 0, ~0, ~0, ~0);
    }
    else if(diff == 6) {
        return _mm256_set_epi32(0, 0, 0, 0, 0, 0, ~0, ~0);
    }
    else if(diff == 7) {
        return _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, ~0);
    }
    else {
        return _mm256_set_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0);    
    }
}

#endif /* common_hpp */
