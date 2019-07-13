//
//  sparse_linear_bin_feature.cpp
//  LinearGBMVector
//


//

#include <stdio.h>
#include "feature.hpp"
#include <x86intrin.h>
#include <omp.h>

SparseLinearBinFeature::SparseLinearBinFeature(DataMat* _train_set, bool _is_categorical,
                                               int _feature_index, BoosterConfig* _booster_config,
                                               const std::vector<int>& _global_leaf_starts, const std::vector<int>& _global_leaf_ends,
                                               int _num_vars, vector<uint8_t>& _data,
                                               vector<int> & _bin_counts, vector<double> & _bin_values):    
LinearBinFeature(_train_set, _is_categorical, _feature_index, _booster_config, _global_leaf_starts, 
                 _global_leaf_ends, _num_vars, _data, _bin_counts, _bin_values) {}

void SparseLinearBinFeature::PrepareHistogram(int leaf_id, bool use_cache, RowHistogram *cur,
                                              RowHistogram *sibling, fvec64 &bin_gradients, bool need_augment) {
    if(num_bins <= 1) {
        return;             
    }
    
    tmp_histograms = cur->Get();
    
    vector<double*> bin_ptrs(num_bins); 
    
    if(leaf_id == 0) {
        prev_num_vars = 0;
        cur_num_vars = 1;
        redundant = false;
        cur->SetDepthAndLeafID(0, leaf_id, need_augment);   
        cur->Clear();
        
        const uint8_t* data_ptr = root_data_ptr->data();//data.data();
        
        for(int i = 0; i < num_bins; ++i) {
            bin_ptrs[i] = (*tmp_histograms)[i].data();  
        }
        
        //vector<dvec32> &histograms = *tmp_histograms;
        float *local_bin_gradients = bin_gradients.data();
        for(int i = 0; i < num_data; i += 4) {
            __m256 __packed_gh = _mm256_load_ps(local_bin_gradients + (i << 1));
            __m256d __gh = _mm256_cvtps_pd(_mm256_extractf128_ps(__packed_gh, 0));
            __m128d __gh0 = _mm256_extractf128_pd(__gh, 0);
            __m128d __gh1 = _mm256_extractf128_pd(__gh, 1);
            
            uint32_t bins = *(uint32_t*)(data_ptr + i);
            
            uint32_t bin = bins & 0xff;
            if(bin != 0) {
                __m128d __hist0 = _mm_load_pd(bin_ptrs[bin]);
                __m128d __result0 = _mm_add_pd(__gh0, __hist0);
                _mm_store_pd(bin_ptrs[bin], __result0);
            }
            
            bin = (bins & 0xff00) >> 8;//data[i + 1];
            if(bin != 0) {
                __m128d __hist1 = _mm_load_pd(bin_ptrs[bin]);
                __m128d __result1 = _mm_add_pd(__gh1, __hist1);
                _mm_store_pd(bin_ptrs[bin], __result1);
            }
            
            __gh = _mm256_cvtps_pd(_mm256_extractf128_ps(__packed_gh, 1));  
            __gh0 = _mm256_extractf128_pd(__gh, 0);
            __gh1 = _mm256_extractf128_pd(__gh, 1);
            
            bin = (bins & 0xff0000) >> 16;//data[i + 2];
            if(bin != 0) {
                __m128d __hist0 = _mm_load_pd(bin_ptrs[bin]);
                __m128d __result0 = _mm_add_pd(__gh0, __hist0);
                _mm_store_pd(bin_ptrs[bin], __result0);
            }
            
            bin = (bins & 0xff000000) >> 24;//data[i + 3];
            if(bin != 0) {
                __m128d __hist1 = _mm_load_pd(bin_ptrs[bin]);
                __m128d __result1 = _mm_add_pd(__gh1, __hist1);
                _mm_store_pd(bin_ptrs[bin], __result1);
            }
        }
        return;
    }
    
    /*redundant = false;
    for(int i = 0; i < last_split_features.size(); ++i) {   
        if(feature_index == last_split_features[i]) {
            redundant = true;
            break;
        }
    }
    
    redundant |= (last_split_features.size() == booster_config->num_vars);
    
    if(!redundant) {        
        cur_num_vars = static_cast<int>(last_split_features.size()) + 1;    
        prev_num_vars = cur_num_vars - 1;
    }
    else {
        cur_num_vars = static_cast<int>(last_split_features.size());
        prev_num_vars = cur_num_vars;
    }*/
    
    int row_size = 2 + prev_num_vars * 3;
    if(prev_num_vars >= 2) {
        row_size += prev_num_vars * (prev_num_vars - 1) / 2;
    }
    
    int leaf_start = global_leaf_starts[leaf_id];
    int leaf_end = global_leaf_ends[leaf_id];
    int leaf_num_data = leaf_end - leaf_start;  
    const uint8_t *local_data_bins = GetLocalData(leaf_id, leaf_start);
    
    if(!use_cache) {
        cur->SetDepthAndLeafID(prev_num_vars, leaf_id, need_augment);                       
        cur->Clear();
        
        float *local_bin_gradient = bin_gradients.data();
        
        for(int i = 0; i < num_bins; ++i) {
            bin_ptrs[i] = (*tmp_histograms)[i].data();
        }
        
        if(row_size <= 8) {
            //row_size == 5
            for(int i = 0; i < leaf_num_data; ++i) {
                uint8_t bin = local_data_bins[i];
                if(bin != 0) {
                    double *local_tmp_histogram = bin_ptrs[bin];
                    __m128 __a = _mm_loadu_ps(local_bin_gradient);
                    __m256d __a0 = _mm256_cvtps_pd(__a);
                    
                    __m256d __b = _mm256_load_pd(local_tmp_histogram);
                    __m256d __c = _mm256_add_pd(__a0, __b);
                    _mm256_store_pd(local_tmp_histogram, __c);
                    
                    local_tmp_histogram[4] += local_bin_gradient[4];        
                }
                local_bin_gradient += 5;                                            
            }
        }
        else if(row_size <= 12) {
            //row_size == 9
            for(int i = 0; i < leaf_num_data; ++i) {
                uint8_t bin = local_data_bins[i];
                if(bin != 0) {
                    double *local_tmp_histogram = bin_ptrs[bin];
                    __m256 __a = _mm256_loadu_ps(local_bin_gradient);
                    __m256d __a0 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 0));
                    __m256d __a1 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 1));
                    
                    __m256d __b = _mm256_load_pd(local_tmp_histogram);
                    __m256d __c = _mm256_add_pd(__a0, __b);
                    _mm256_store_pd(local_tmp_histogram, __c);
                    
                    __b = _mm256_load_pd(local_tmp_histogram + 4);
                    __c = _mm256_add_pd(__a1, __b);
                    _mm256_store_pd(local_tmp_histogram + 4, __c);
                    
                    local_tmp_histogram[8] += local_bin_gradient[8];
                }
                local_bin_gradient += 9;
            }
        }
        else if(row_size <= 16) {
            //row_size == 14
            for(int i = 0; i < leaf_num_data; ++i) {
                uint8_t bin = local_data_bins[i];
                if(bin != 0) {
                    double *local_tmp_histogram = bin_ptrs[bin];
                    __m256 __a = _mm256_loadu_ps(local_bin_gradient);
                    __m256d __a0 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 0));
                    __m256d __a1 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 1));
                    
                    __m256d __b = _mm256_load_pd(local_tmp_histogram);
                    __m256d __c = _mm256_add_pd(__a0, __b);
                    _mm256_store_pd(local_tmp_histogram, __c);
                    
                    __b = _mm256_load_pd(local_tmp_histogram + 4);
                    __c = _mm256_add_pd(__a1, __b);
                    _mm256_store_pd(local_tmp_histogram + 4, __c);
                    
                    __a = _mm256_loadu_ps(local_bin_gradient + 8);
                    __a0 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 0));
                    __a1 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 1));
                    
                    __b = _mm256_load_pd(local_tmp_histogram + 8);
                    __c = _mm256_add_pd(__a0, __b);
                    _mm256_store_pd(local_tmp_histogram + 8, __c);
                    
                    __b = _mm256_load_pd(local_tmp_histogram + 12);
                    __c = _mm256_add_pd(__a1, __b);
                    _mm256_store_pd(local_tmp_histogram + 12, __c);
                }
                
                local_bin_gradient += 14;
            }
        }
        else if(row_size <= 20) {
            //row_size == 20
            for(int i = 0; i < leaf_num_data; ++i) {
                uint8_t bin = local_data_bins[i];
                if(bin != 0) {
                    double *local_tmp_histogram = bin_ptrs[bin];
                    __m256 __a = _mm256_loadu_ps(local_bin_gradient);
                    __m256d __a0 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 0));
                    __m256d __a1 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 1));
                    
                    __m256d __b = _mm256_load_pd(local_tmp_histogram);
                    __m256d __c = _mm256_add_pd(__a0, __b);
                    _mm256_store_pd(local_tmp_histogram, __c);
                    
                    __b = _mm256_load_pd(local_tmp_histogram + 4);
                    __c = _mm256_add_pd(__a1, __b);
                    _mm256_store_pd(local_tmp_histogram + 4, __c);
                    
                    __a = _mm256_loadu_ps(local_bin_gradient + 8);
                    __a0 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 0));
                    __a1 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 1));
                    
                    __b = _mm256_load_pd(local_tmp_histogram + 8);
                    __c = _mm256_add_pd(__a0, __b);
                    _mm256_store_pd(local_tmp_histogram + 8, __c);
                    
                    __b = _mm256_load_pd(local_tmp_histogram + 12);
                    __c = _mm256_add_pd(__a1, __b);
                    _mm256_store_pd(local_tmp_histogram + 12, __c);
                    
                    __m128 __aa = _mm_loadu_ps(local_bin_gradient + 16);
                    __a0 = _mm256_cvtps_pd(__aa);
                    __b = _mm256_load_pd(local_tmp_histogram + 16);
                    __c = _mm256_add_pd(__a0, __b);
                    _mm256_store_pd(local_tmp_histogram + 16, __c);
                }
                
                local_bin_gradient += 20;
            }
        }
        else if(row_size <= 28) {
            //row_size == 27
            for(int i = 0; i < leaf_num_data; ++i) {
                uint8_t bin = local_data_bins[i];
                if(bin != 0) {
                    double *local_tmp_histogram = bin_ptrs[bin];
                    __m256 __a = _mm256_loadu_ps(local_bin_gradient);
                    __m256d __a0 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 0));
                    __m256d __a1 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 1));
                    
                    __m256d __b = _mm256_load_pd(local_tmp_histogram);
                    __m256d __c = _mm256_add_pd(__a0, __b);
                    _mm256_store_pd(local_tmp_histogram, __c);
                    
                    __b = _mm256_load_pd(local_tmp_histogram + 4);
                    __c = _mm256_add_pd(__a1, __b);
                    _mm256_store_pd(local_tmp_histogram + 4, __c);
                    
                    __a = _mm256_loadu_ps(local_bin_gradient + 8);
                    __a0 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 0));
                    __a1 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 1));
                    
                    __b = _mm256_load_pd(local_tmp_histogram + 8);
                    __c = _mm256_add_pd(__a0, __b);
                    _mm256_store_pd(local_tmp_histogram + 8, __c);
                    
                    __b = _mm256_load_pd(local_tmp_histogram + 12);
                    __c = _mm256_add_pd(__a1, __b);
                    _mm256_store_pd(local_tmp_histogram + 12, __c);
                    
                    __a = _mm256_loadu_ps(local_bin_gradient + 16);
                    __a0 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 0));
                    __a1 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 1));
                    
                    __b = _mm256_load_pd(local_tmp_histogram + 16);
                    __c = _mm256_add_pd(__a0, __b);
                    _mm256_store_pd(local_tmp_histogram + 16, __c);
                    
                    __b = _mm256_load_pd(local_tmp_histogram + 20);
                    __c = _mm256_add_pd(__a1, __b);
                    _mm256_store_pd(local_tmp_histogram + 20, __c);
                    
                    __m128 __aa = _mm_loadu_ps(local_bin_gradient + 24);
                    __a0 = _mm256_cvtps_pd(__aa);
                    __b = _mm256_load_pd(local_tmp_histogram + 24);
                    __c = _mm256_add_pd(__a0, __b);
                    _mm256_store_pd(local_tmp_histogram + 24, __c);
                }
                
                local_bin_gradient += 27;
            }
        }
        else {
            for(int i = 0; i < leaf_num_data; ++i) {
                uint8_t bin = local_data_bins[i];
                if(bin != 0) {
                    double *local_tmp_histogram = bin_ptrs[bin];
                    for(int j = 0; j < row_size; j += 8) {
                        __m256 __a = _mm256_loadu_ps(local_bin_gradient + j);
                        
                        __m256d __a0 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 0));
                        __m256d __a1 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 1));
                        
                        __m256d __b = _mm256_load_pd(local_tmp_histogram + j);
                        __m256d __c = _mm256_add_pd(__a0, __b);
                        _mm256_store_pd(local_tmp_histogram + j, __c);
                        
                        __b = _mm256_load_pd(local_tmp_histogram + j + 4);
                        __c = _mm256_add_pd(__a1, __b);
                        _mm256_store_pd(local_tmp_histogram + j + 4, __c);
                    }
                }
                local_bin_gradient += row_size;
            }
        }
    }
    else {
        cur->Substract(sibling);
        
        cur->SetDepthAndLeafID(prev_num_vars, leaf_id, need_augment);
        
        if(need_augment) {
            int offset = 2 + 3 * (prev_num_vars - 1) + (prev_num_vars - 1) * (prev_num_vars - 2) / 2;
            for(int i = 0; i < num_bins; ++i) {
                bin_ptrs[i] = (*tmp_histograms)[i].data() + offset;
            }
            row_size = prev_num_vars + 2;
            for(int i = 0; i < num_bins; ++i) {
                double *local_tmp_histogram = bin_ptrs[i];                  
#pragma omp simd
                for(int j = 0; j < prev_num_vars + 2; ++j) {
                    local_tmp_histogram[j] = 0.0;
                }
            }
            
            float *local_bin_gradient = bin_gradients.data();
            if(row_size <= 4) {
                for(int i = 0; i < leaf_num_data; ++i) {
                    uint8_t bin = local_data_bins[i];
                    if(bin != 0) {
                        double *local_tmp_histogram = bin_ptrs[bin];
                        __m256 __a = _mm256_loadu_ps(local_bin_gradient);
                        __m256d __a0 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 0));
                        
                        __m256d __b = _mm256_loadu_pd(local_tmp_histogram);
                        __m256d __c = _mm256_add_pd(__a0, __b);
                        _mm256_storeu_pd(local_tmp_histogram, __c);
                    }
                    local_bin_gradient += row_size;
                }
            }
            else if(row_size == 5) {
                for(int i = 0; i < leaf_num_data; ++i) {
                    uint8_t bin = local_data_bins[i];
                    if(bin != 0) {
                        double *local_tmp_histogram = bin_ptrs[bin];
                        __m128 __a = _mm_loadu_ps(local_bin_gradient);
                        __m256d __a0 = _mm256_cvtps_pd(__a);
                        
                        __m256d __b = _mm256_loadu_pd(local_tmp_histogram);
                        __m256d __c = _mm256_add_pd(__a0, __b);
                        _mm256_storeu_pd(local_tmp_histogram, __c);
                        
                        local_tmp_histogram[4] += local_bin_gradient[4];
                    }
                    local_bin_gradient += row_size;
                }
            }
            else if(row_size <= 8) {
                for(int i = 0; i < leaf_num_data; ++i) {
                    uint8_t bin = local_data_bins[i];
                    if(bin != 0) {
                        double *local_tmp_histogram = bin_ptrs[bin];
                        __m256 __a = _mm256_loadu_ps(local_bin_gradient);
                        __m256d __a0 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 0));
                        __m256d __a1 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 1));
                        
                        __m256d __b = _mm256_loadu_pd(local_tmp_histogram);
                        __m256d __c = _mm256_add_pd(__a0, __b);
                        _mm256_storeu_pd(local_tmp_histogram, __c);
                        
                        __b = _mm256_loadu_pd(local_tmp_histogram + 4);
                        __c = _mm256_add_pd(__a1, __b);
                        _mm256_storeu_pd(local_tmp_histogram + 4, __c);
                    }
                    local_bin_gradient += row_size;
                }
            }
            else if(row_size == 9) {
                for(int i = 0; i < leaf_num_data; ++i) {
                    uint8_t bin = local_data_bins[i];
                    if(bin != 0) {
                        double *local_tmp_histogram = bin_ptrs[bin];
                        __m256 __a = _mm256_loadu_ps(local_bin_gradient);
                        __m256d __a0 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 0));
                        __m256d __a1 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 1));
                        
                        __m256d __b = _mm256_loadu_pd(local_tmp_histogram);
                        __m256d __c = _mm256_add_pd(__a0, __b);
                        _mm256_storeu_pd(local_tmp_histogram, __c);
                        
                        __b = _mm256_loadu_pd(local_tmp_histogram + 4);
                        __c = _mm256_add_pd(__a1, __b);
                        _mm256_storeu_pd(local_tmp_histogram + 4, __c);
                        
                        local_tmp_histogram[8] += local_bin_gradient[8];
                    }
                    local_bin_gradient += row_size;
                }
            }
            else if(row_size <= 12) {
                for(int i = 0; i < leaf_num_data; ++i) {
                    uint8_t bin = local_data_bins[i];
                    if(bin != 0) {
                        double *local_tmp_histogram = bin_ptrs[bin];
                        __m256 __a = _mm256_loadu_ps(local_bin_gradient);
                        __m256d __a0 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 0));
                        __m256d __a1 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 1));
                        
                        __m256d __b = _mm256_loadu_pd(local_tmp_histogram);
                        __m256d __c = _mm256_add_pd(__a0, __b);
                        _mm256_storeu_pd(local_tmp_histogram, __c);
                        
                        __b = _mm256_loadu_pd(local_tmp_histogram + 4);
                        __c = _mm256_add_pd(__a1, __b);
                        _mm256_storeu_pd(local_tmp_histogram + 4, __c);
                        
                        __m128 __as = _mm_loadu_ps(local_bin_gradient + 8);
                        __a0 = _mm256_cvtps_pd(__as);
                        
                        __b = _mm256_loadu_pd(local_tmp_histogram + 8);
                        __c = _mm256_add_pd(__a0, __b);
                        _mm256_storeu_pd(local_tmp_histogram + 8, __c);
                    }
                    
                    local_bin_gradient += row_size;
                }
            }
            else if(row_size == 13) {
                for(int i = 0; i < leaf_num_data; ++i) {
                    uint8_t bin = local_data_bins[i];
                    if(bin != 0) {
                        double *local_tmp_histogram = bin_ptrs[bin];
                        __m256 __a = _mm256_loadu_ps(local_bin_gradient);
                        __m256d __a0 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 0));
                        __m256d __a1 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 1));
                        
                        __m256d __b = _mm256_loadu_pd(local_tmp_histogram);
                        __m256d __c = _mm256_add_pd(__a0, __b);
                        _mm256_storeu_pd(local_tmp_histogram, __c);
                        
                        __b = _mm256_loadu_pd(local_tmp_histogram + 4);
                        __c = _mm256_add_pd(__a1, __b);
                        _mm256_storeu_pd(local_tmp_histogram + 4, __c);
                        
                        __m128 __as = _mm_loadu_ps(local_bin_gradient + 8);
                        __a0 = _mm256_cvtps_pd(__as);
                        
                        __b = _mm256_loadu_pd(local_tmp_histogram + 8);
                        __c = _mm256_add_pd(__a0, __b);
                        _mm256_storeu_pd(local_tmp_histogram + 8, __c);
                        
                        local_tmp_histogram[12] += local_bin_gradient[12];
                    }
                    
                    local_bin_gradient += row_size;
                }
            }
            else {
                for(int i = 0; i < leaf_num_data; ++i) {
                    uint8_t bin = local_data_bins[i];
                    if(bin != 0) {
                        double *local_tmp_histogram = bin_ptrs[bin];
                        for(int j = 0; j < row_size; j += 8) {
                            __m256 __a = _mm256_loadu_ps(local_bin_gradient + j);
                            __m256d __a0 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 0));  
                            __m256d __a1 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 1));
                            
                            __m256d __b = _mm256_loadu_pd(local_tmp_histogram + j);
                            __m256d __c = _mm256_add_pd(__a0, __b);
                            _mm256_storeu_pd(local_tmp_histogram + j, __c);
                            
                            __b = _mm256_loadu_pd(local_tmp_histogram + j + 4);
                            __c = _mm256_add_pd(__a1, __b);
                            _mm256_storeu_pd(local_tmp_histogram + j + 4, __c);
                        }
                    }
                    local_bin_gradient += row_size;
                }
            }
        }
    }
}

SplitInfo* SparseLinearBinFeature::FindBestSplit(int leaf_id, double leaf_gain,
                                                 const dvec64 &leaf_sum_up,
                                                 const vector<int> &last_split_features) {
    if(num_bins <= 1) {
        return nullptr;
    }
    
    double l1_reg = booster_config->l2_reg; 
    
    int row_size = 2 + prev_num_vars * 3;
    if(prev_num_vars >= 2) {
        row_size += prev_num_vars * (prev_num_vars - 1) / 2;
    }
    
    dvec64 best_left_k, best_right_k;
    
    double best_left_gain = 0.0, best_right_gain = 0.0;
    
    double best_gain = 0.0;
    
    uint8_t best_split_point = 0;
    
    dvec64 left_k(cur_num_vars + 1, 0.0), right_k(cur_num_vars + 1, 0.0);
    
    const vector<dvec32> &histograms = *tmp_histograms;
    double t_start = omp_get_wtime();
    if(!redundant) {
        dvec32 left_histogram(row_size, 0.0);
        dvec32 right_histogram(row_size, 0.0);
        double *left_histogram_ptr = left_histogram.data(), *right_histogram_ptr = right_histogram.data();
        const double *sum_up_ptr = leaf_sum_up.data();
        //initialize right histogram
#pragma omp simd aligned(left_histogram_ptr,sum_up_ptr:32)
        for(int i = 0; i < row_size; ++i) {
            left_histogram_ptr[i] = sum_up_ptr[i];
        }
        right_matrix.clear();
        left_matrix.clear();
        right_matrix.resize(row_size + 1, 0.0);
        left_matrix.resize(row_size + 1, 0.0);
        left_vec.clear();
        left_vec.resize(2 + prev_num_vars, 0.0);
        right_vec.clear();
        right_vec.resize(2 + prev_num_vars, 0.0);
        left_matrix.back() = booster_config->l2_reg;
        right_matrix.back() = booster_config->l2_reg;
        int matrix_base = row_size - 1 - prev_num_vars;
        //start from 1 here. need to be modified when using mean values in a bin as bin value
        for(int i = 0; i < num_bins; ++i) {
            double bin_value = bin_values[i];
            const dvec32 &bin_histogram = histograms[i];
            double hessian = bin_histogram[1];
            left_vec[prev_num_vars + 1] -= bin_value * bin_histogram[0];
            left_matrix[matrix_base] += bin_value * hessian;
#pragma omp simd
            for(int j = 0; j < prev_num_vars; ++j) {
                left_matrix[matrix_base + 1 + j] += bin_value * bin_histogram[3 + 3 * j + j * (j - 1) / 2];
            }
            left_matrix[matrix_base + 1 + prev_num_vars] += bin_value * bin_value * hessian;
        }
        
        for(int i = num_bins - 2; i >= 0; --i) {
            const double *bin_histogram = histograms[i + 1].data();
            const double bin_value = bin_values[i + 1];
            const double gradient = bin_histogram[0];
            const double hessian = bin_histogram[1];
            
            if(hessian == 0.0) {
                continue;
            }
            
#pragma omp simd aligned(right_histogram_ptr,left_histogram_ptr,bin_histogram:32)
            for(int j = 0; j < row_size; ++j) {
                left_histogram_ptr[j] -= bin_histogram[j];
                right_histogram_ptr[j] += bin_histogram[j];
            }
            
            LoadMatrix(left_histogram, prev_num_vars, left_matrix, left_vec);
            LoadMatrix(right_histogram, prev_num_vars, right_matrix, right_vec);
            
            double value = bin_value * gradient;
            right_vec[prev_num_vars + 1] -= value;
            left_vec[prev_num_vars + 1] += value;
            value = bin_value * hessian;
            right_matrix[matrix_base] += value;
            left_matrix[matrix_base] -= value;
            
            value *= bin_value;
            right_matrix[matrix_base + 1 + prev_num_vars] += value;
            left_matrix[matrix_base + 1 + prev_num_vars] -= value;
#pragma omp simd
            for(int j = 0; j < prev_num_vars; ++j) {
                value = bin_value * bin_histogram[3 + 3 * j + j * (j - 1) / 2];
                right_matrix[matrix_base + 1 + j] += value;
                left_matrix[matrix_base + 1 + j] -= value;
            }
            
            if(right_matrix[0] < booster_config->min_sum_hessian_in_leaf +
               booster_config->l2_reg) continue;
            
            if(left_matrix[0] < booster_config->min_sum_hessian_in_leaf +
               booster_config->l2_reg) break;
            
            if(leaf_id == 1 && feature_index == 116 && i == 18) {
                
            }
            
            double left_gain = Solve(left_matrix, left_vec, left_k, cur_num_vars + 1, l1_reg, row_size + 1);
            
            double right_gain = Solve(right_matrix, right_vec, right_k, cur_num_vars + 1, l1_reg, row_size + 1);
            
            double gain = leaf_gain - left_gain - right_gain;
            
            if(gain > best_gain) {
                best_left_gain = left_gain;
                best_right_gain = right_gain;
                best_left_k = left_k;
                best_right_k = right_k;
                
                best_gain = gain;
                
                best_split_point = i;
            }
        }
    }
    else {
        right_matrix.clear();
        left_matrix.clear();
        right_matrix.resize(row_size - 1 - prev_num_vars, 0.0);
        left_matrix.resize(row_size - 1 - prev_num_vars, 0.0);
        left_vec.clear();
        left_vec.resize(1 + prev_num_vars, 0.0);
        right_vec.clear();
        right_vec.resize(1 + prev_num_vars, 0.0);
        
        dvec32 left_histogram(row_size, 0.0), right_histogram(row_size, 0.0);
        double *left_histogram_ptr = left_histogram.data(), *right_histogram_ptr = right_histogram.data();
        const double *sum_up_data = leaf_sum_up.data();
        
#pragma omp simd aligned(left_histogram_ptr,sum_up_data:32)
        for(int i = 0; i < row_size; ++i) {
            left_histogram_ptr[i] = sum_up_data[i];             
        }
        
        for(int i = num_bins - 2; i >= 0; --i) {
            
            const double *bin_histogram = histograms[i + 1].data();
            
            if(bin_histogram[1] == 0.0) {
                continue;
            }
            
#pragma omp simd aligned(right_histogram_ptr,left_histogram_ptr,bin_histogram:32)
            for(int i = 0; i < row_size; ++i) {
                right_histogram_ptr[i] += bin_histogram[i];
                left_histogram_ptr[i] -= bin_histogram[i];
            }
            
            
            LoadMatrix(left_histogram, prev_num_vars, left_matrix, left_vec);
            LoadMatrix(right_histogram, prev_num_vars, right_matrix, right_vec);            
            
            if(right_matrix[0] < booster_config->min_sum_hessian_in_leaf +
               booster_config->l2_reg) continue;
            
            if(left_matrix[0] < booster_config->min_sum_hessian_in_leaf +
               booster_config->l2_reg) break;   
            
            double left_gain = Solve(left_matrix, left_vec, left_k,
                                     cur_num_vars + 1, l1_reg, row_size - 1 - prev_num_vars);
            
            double right_gain = Solve(right_matrix, right_vec, right_k,
                                      cur_num_vars + 1, l1_reg, row_size - 1 - prev_num_vars);
            
            double gain = leaf_gain - left_gain - right_gain;
            
            if(gain > best_gain) {
                best_left_gain = left_gain;
                best_right_gain = right_gain;
                best_left_k = left_k;
                best_right_k = right_k;
                
                best_gain = gain;
                
                best_split_point = i;
            }
        }
    }
    
    if(best_gain > booster_config->min_gain + 0.001) {
        vector<int> split_features;
        for(int i = static_cast<int>(last_split_features.size()) - prev_num_vars;
            i < static_cast<int>(last_split_features.size()); ++i) {
            split_features.push_back(last_split_features[i]);
        }
        
        if(!redundant) {
            split_features.push_back(feature_index);
        }
        
        double best_left_b = 0.0, best_right_b = 0.0;
        vector<double> best_left_ks(cur_num_vars, 0.0), best_right_ks(cur_num_vars, 0.0);
        best_left_b = best_left_k.front();
        best_right_b = best_right_k.front();
#pragma omp simd
        for(int i = 1; i < cur_num_vars + 1; ++i) {
            best_left_ks[i - 1] = best_left_k[i];
            best_right_ks[i - 1] = best_right_k[i];
        }
        
        SplitInfo *split = new MultipleLinearSplitInfo(feature_index,
                                                       best_split_point,
                                                       bin_boundaries[best_split_point],
                                                       best_gain,
                                                       best_left_gain,
                                                       best_right_gain,
                                                       best_left_ks,
                                                       best_left_b,
                                                       best_right_ks,
                                                       best_right_b,
                                                       split_features);
        double t_end = omp_get_wtime();
        load_matrix_time += (t_end - t_start);
        return split;
    }
    double t_end = omp_get_wtime();
    load_matrix_time += (t_end - t_start);
    
    return nullptr;
}
