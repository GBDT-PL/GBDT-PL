//
//  row_histogram.cpp
//  LinearGBMVector
//


//

#include "row_histogram.hpp"
#include <x86intrin.h>

void RowHistogram::Substract(RowHistogram *child) {
    vector<dvec32> &child_histograms = *child->Get();
    vector<dvec32> &histograms = *sub_histograms;
    
    if(cur_var < child->cur_var || (cur_var == child->cur_var && !child->need_augment)) {
        int row_size = 2 + 3 * cur_var;
        if(cur_var >= 2) {
            row_size += cur_var * (cur_var - 1) / 2;
        }
        for(int i = 0; i < max_bin; ++i) {
            double* local_child = child_histograms[i].data();
            double* local = histograms[i].data();
            __m256d __a;
            __m256d __b;
            __m256d __c;
            for(int j = 0; j < row_size; j += 4) {
                __a = _mm256_load_pd(local_child + j);              
                __b = _mm256_load_pd(local + j);
                __c = _mm256_sub_pd(__b, __a);
                _mm256_store_pd(local + j, __c);
            }
        }
    }
    else {
        /*for(int i = 0; i < max_bin; ++i) {
            double *local_child = child_histograms[i].data();       
            double *local = histograms[i].data() + 5;
            double *local_output = histograms[i].data();
            local_output[0] = histograms[i][0] - local_child[0];
            local_output[1] = histograms[i][1] - local_child[1];
            local_child += 2;
            local_output += 2;
            for(int j = 0; j < max_var - 2; ++j) {
                local_output[0] = local[0] - local_child[0];
                local_output[1] = local[1] - local_child[1];
                local_output[2] = local[2] - local_child[2]; 
#pragma omp simd	
                for(int k = 0; k < j; ++k) {
                    local_output[3 + k] = local[4 + k] - local_child[3 + k];    
                }
                local_output += 3 + j;
                local_child += 3 + j;
                local += 4 + j;
            }   
        }*/
        
        //for additive linear node only
        for(int i = 0; i < max_bin; ++i) {
            double *local_child = child_histograms[i].data();
            double *local = histograms[i].data();   
#pragma omp simd
            for(int j = 0; j < 2; ++j) {
                local[j] -= local_child[j]; 
            }
        }
    }
}
