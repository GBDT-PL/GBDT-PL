//
//  linear_bin_feature.cpp
//  LinearGBM
//


//

#include "feature.hpp"
#include <cstring>
#include <vector>
#include <cmath>
#include <fstream>
#include <omp.h>    
#include <mkl.h>
#include <x86intrin.h>

LinearBinFeature::LinearBinFeature(DataMat* _train_set, bool _is_categorical,
                                   int _feature_index, BoosterConfig* _booster_config,
                                   const std::vector<int> &_global_leaf_starts,
                                   const std::vector<int> &_global_leaf_ends,
                                   int _num_vars, vector<uint8_t>& _data,
                                   vector<int> & _bin_counts, vector<double> & _bin_values):
OrderedFeature(_train_set, _is_categorical, _feature_index, _booster_config),
global_leaf_starts(_global_leaf_starts), global_leaf_ends(_global_leaf_ends),
bin_boundaries(_train_set->bin_boundaries[_feature_index]), data(_data), bin_counts(_bin_counts),
bin_values(_bin_values) {
    train_set = _train_set;
    num_bins = train_set->num_bins_per_feature[feature_index];
    bin_counts.resize(num_bins, 0);
    bin_values.resize(num_bins, 0.0);
    data.resize(num_data);
    ordered_data.clear();
    ordered_data.shrink_to_fit();
    
    tmp_histograms = nullptr; 
    
    left_matrix.resize((_num_vars + 1) * (_num_vars + 1) + 4, 0.0);
    left_vec.resize(_num_vars + 1, 0.0);
    right_matrix.resize((_num_vars + 1) * (_num_vars + 1) + 4, 0.0);
    right_vec.resize(_num_vars + 1, 0.0);
    
    matrix_copy = (double *)mkl_calloc((_num_vars + 1) * (_num_vars + 1) + 4, sizeof(double), 64);
    
    mkl_info = 0;
    mkl_m = 1;
    mkl_ipiv = new int[_num_vars + 1];
    
    mkl_time = 0.0;
    load_matrix_time = 0.0;
    
    root_data_ptr = &data;
    
    //leaf_local_data.resize(2 * booster_config->max_leaf + 1, vector<uint8_t>());
    leaf_local_data.resize(num_data + 8, 0);
    copy_tmp.resize(num_data + 16, 0);
    if(num_bins >= 2) {
        useful = true;  
    }
    else {
        useful = false;
    }
}

void LinearBinFeature::PushIntoBin(double fvalue, int data_idx) {
    if(num_bins <= 1) {
        data[data_idx] = 0; 
        return; 
    }
    
    int low = 0, high = num_bins, mid = (low + high) / 2;
    
    while(!((mid >= num_bins - 1 && bin_boundaries.back() < fvalue) ||
            (mid <= 0 && bin_boundaries[0] >= fvalue)  ||
            (bin_boundaries[mid - 1] < fvalue && bin_boundaries[mid] >= fvalue)) && low < high) {
        if(bin_boundaries[mid - 1] >= fvalue) {
            high = mid;
        }
        else {
            low = mid + 1;
        }
        mid = (low + high) / 2; 
    }
    if(mid <= 0) {
        data[data_idx] = 0; 
    }
    else if(mid >= num_bins - 1) {
        data[data_idx] = num_bins - 1;
    }
    else {
        data[data_idx] = static_cast<uint8_t>(mid);
    }
    ++bin_counts[mid];
    bin_values[mid] += fvalue;  
}

void LinearBinFeature::PrepareHistogram(int leaf_id, bool use_cache,
                                        RowHistogram *cur,
                                        RowHistogram *sibling,
                                        fvec64& bin_gradients,
                                        bool need_augment) {
    tmp_histograms = nullptr; 
    
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
        
        const uint8_t* data_ptr = root_data_ptr->data();
        
        for(int i = 0; i < num_bins; ++i) {
            bin_ptrs[i] = (*tmp_histograms)[i].data();
        }
        
        float *local_bin_gradient = bin_gradients.data();   
	int num_data_aligned = num_data / 4 * 4;
        for(int i = 0; i < num_data_aligned; i += 4) {
            __m256 __packed_gh = _mm256_load_ps(local_bin_gradient + (i << 1));
            __m256d __gh = _mm256_cvtps_pd(_mm256_extractf128_ps(__packed_gh, 0));          
            __m128d __gh0 = _mm256_extractf128_pd(__gh, 0);
            __m128d __gh1 = _mm256_extractf128_pd(__gh, 1);
            
            uint32_t bins = *(uint32_t*)(data_ptr + i);
            
            uint32_t bin = bins & 0xff;
            __m128d __hist0 = _mm_load_pd(bin_ptrs[bin]);
            __m128d __result0 = _mm_add_pd(__gh0, __hist0); 
            _mm_store_pd(bin_ptrs[bin], __result0);
            
            bin = (bins & 0xff00) >> 8;//data[i + 1];
            __m128d __hist1 = _mm_load_pd(bin_ptrs[bin]);
            __m128d __result1 = _mm_add_pd(__gh1, __hist1);
            _mm_store_pd(bin_ptrs[bin], __result1);
            
            __gh = _mm256_cvtps_pd(_mm256_extractf128_ps(__packed_gh, 1));
            __gh0 = _mm256_extractf128_pd(__gh, 0);
            __gh1 = _mm256_extractf128_pd(__gh, 1);
            
            bin = (bins & 0xff0000) >> 16;//data[i + 2];
            __hist0 = _mm_load_pd(bin_ptrs[bin]);
            __result0 = _mm_add_pd(__gh0, __hist0);
            _mm_store_pd(bin_ptrs[bin], __result0);
            
            bin = (bins & 0xff000000) >> 24;//data[i + 3];                        
            __hist1 = _mm_load_pd(bin_ptrs[bin]);
            __result1 = _mm_add_pd(__gh1, __hist1);
            _mm_store_pd(bin_ptrs[bin], __result1);
        }
	for(int i = num_data_aligned; i < num_data; ++i) {
	    uint8_t bin = data_ptr[i];
	    bin_ptrs[bin][0] += local_bin_gradient[2 * i];
	    bin_ptrs[bin][1] += local_bin_gradient[2 * i + 1];
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
                double *local_tmp_histogram = bin_ptrs[local_data_bins[i]];         
                __m128 __a = _mm_loadu_ps(local_bin_gradient);
                __m256d __a0 = _mm256_cvtps_pd(__a);
                
                __m256d __b = _mm256_load_pd(local_tmp_histogram);
                __m256d __c = _mm256_add_pd(__a0, __b);
                _mm256_store_pd(local_tmp_histogram, __c);
                
                local_tmp_histogram[4] += local_bin_gradient[4];
                local_bin_gradient += 5;
            }
        }
        else if(row_size <= 12) {
            //row_size == 9
            for(int i = 0; i < leaf_num_data; ++i) {
                double *local_tmp_histogram = bin_ptrs[local_data_bins[i]];
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
                
                local_bin_gradient += 9;
            }
        }
        else if(row_size <= 16) {
            //row_size == 14
            for(int i = 0; i < leaf_num_data; ++i) {
                double *local_tmp_histogram = bin_ptrs[local_data_bins[i]];
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
                
                local_bin_gradient += 14;
            }
        }
        else if(row_size <= 20) {
            //row_size == 20
            for(int i = 0; i < leaf_num_data; ++i) {
                double *local_tmp_histogram = bin_ptrs[local_data_bins[i]];
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
                
                local_bin_gradient += 20;
            }
        }
        else if(row_size <= 28) {
            //row_size == 27
            for(int i = 0; i < leaf_num_data; ++i) {
                double *local_tmp_histogram = bin_ptrs[local_data_bins[i]];
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
                
                local_bin_gradient += 27;
            }
        }
        else {
            for(int i = 0; i < leaf_num_data; ++i) {
                double *local_tmp_histogram = bin_ptrs[local_data_bins[i]];
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
                    double *local_tmp_histogram = bin_ptrs[local_data_bins[i]]; 
                    __m256 __a = _mm256_loadu_ps(local_bin_gradient);
                    __m256d __a0 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 0));
                        
                    __m256d __b = _mm256_loadu_pd(local_tmp_histogram);
                    __m256d __c = _mm256_add_pd(__a0, __b);
                    _mm256_storeu_pd(local_tmp_histogram, __c);
                    local_bin_gradient += row_size;
                }
            }
            else if(row_size == 5) {
                for(int i = 0; i < leaf_num_data; ++i) {
                    double *local_tmp_histogram = bin_ptrs[local_data_bins[i]];
                    __m128 __a = _mm_loadu_ps(local_bin_gradient);
                    __m256d __a0 = _mm256_cvtps_pd(__a);
                    
                    __m256d __b = _mm256_loadu_pd(local_tmp_histogram);
                    __m256d __c = _mm256_add_pd(__a0, __b);
                    _mm256_storeu_pd(local_tmp_histogram, __c);
                    
                    local_tmp_histogram[4] += local_bin_gradient[4];
                    local_bin_gradient += row_size;
                }
            }
            else if(row_size <= 8) {
                for(int i = 0; i < leaf_num_data; ++i) {
                    double *local_tmp_histogram = bin_ptrs[local_data_bins[i]];
                    __m256 __a = _mm256_loadu_ps(local_bin_gradient);
                    __m256d __a0 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 0));
                    __m256d __a1 = _mm256_cvtps_pd(_mm256_extractf128_ps(__a, 1));
                    
                    __m256d __b = _mm256_loadu_pd(local_tmp_histogram);
                    __m256d __c = _mm256_add_pd(__a0, __b);
                    _mm256_storeu_pd(local_tmp_histogram, __c);
                    
                    __b = _mm256_loadu_pd(local_tmp_histogram + 4);
                    __c = _mm256_add_pd(__a1, __b);
                    _mm256_storeu_pd(local_tmp_histogram + 4, __c);
                    local_bin_gradient += row_size;
                }
            }
            else if(row_size == 9) {
                for(int i = 0; i < leaf_num_data; ++i) {
                    double *local_tmp_histogram = bin_ptrs[local_data_bins[i]];
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
                    local_bin_gradient += row_size;
                }
            }
            else if(row_size <= 12) {
                for(int i = 0; i < leaf_num_data; ++i) {
                    double *local_tmp_histogram = bin_ptrs[local_data_bins[i]];
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
                    
                    local_bin_gradient += row_size;
                }
            }
            else if(row_size == 13) {
                for(int i = 0; i < leaf_num_data; ++i) {
                    double *local_tmp_histogram = bin_ptrs[local_data_bins[i]];
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
                    
                    local_bin_gradient += row_size;
                }
            }
            else {
                for(int i = 0; i < leaf_num_data; ++i) {
                    double *local_tmp_histogram = bin_ptrs[local_data_bins[i]];
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
                    local_bin_gradient += row_size;
                }
            }
        }
    }
}

SplitInfo* LinearBinFeature::FindBestSplit(int leaf_id,
                                           double leaf_gain,
                                           const dvec64 &leaf_sum_up,
                                           const vector<int>& last_split_features) {
    //Sweep all the split points in order of the feature value on the leaf and find the best one.
    if(num_bins <= 1) {
        return nullptr;
    }
    
    double l3_reg = booster_config->l2_reg;
    
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
#pragma omp simd aligned(right_histogram_ptr,sum_up_ptr:32)
        for(int i = 0; i < row_size; ++i) {
            right_histogram_ptr[i] = sum_up_ptr[i];
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
            right_vec[prev_num_vars + 1] -= bin_value * bin_histogram[0];
            right_matrix[matrix_base] += bin_value * hessian;
#pragma omp simd
            for(int j = 0; j < prev_num_vars; ++j) {
                right_matrix[matrix_base + 1 + j] += bin_value * bin_histogram[3 + 3 * j + j * (j - 1) / 2];
            }
            right_matrix[matrix_base + 1 + prev_num_vars] += bin_value * bin_value * hessian;
        }
        
        for(int i = 0; i < num_bins - 1; ++i) {
            const double *bin_histogram = histograms[i].data();
            const double bin_value = bin_values[i];
            const double gradient = bin_histogram[0];
            const double hessian = bin_histogram[1];
            
            if(hessian == 0.0) {
                continue;
            }
            
#pragma omp simd aligned(right_histogram_ptr,left_histogram_ptr,bin_histogram:32)
            for(int j = 0; j < row_size; ++j) {
                left_histogram_ptr[j] += bin_histogram[j];
                right_histogram_ptr[j] -= bin_histogram[j];
            }
            
            LoadMatrix(left_histogram, prev_num_vars, left_matrix, left_vec);
            LoadMatrix(right_histogram, prev_num_vars, right_matrix, right_vec);
            
            double value = bin_value * gradient;
            right_vec[prev_num_vars + 1] += value;
            left_vec[prev_num_vars + 1] -= value;
            value = bin_value * hessian;
            right_matrix[matrix_base] -= value;
            left_matrix[matrix_base] += value;
            
            value *= bin_value;
            right_matrix[matrix_base + 1 + prev_num_vars] -= value; 
            left_matrix[matrix_base + 1 + prev_num_vars] += value;
#pragma omp simd
            for(int j = 0; j < prev_num_vars; ++j) {
                value = bin_value * bin_histogram[3 + 3 * j + j * (j - 1) / 2];
                right_matrix[matrix_base + 1 + j] -= value;
                left_matrix[matrix_base + 1 + j] += value;
            }
            
            if(left_matrix[0] < booster_config->min_sum_hessian_in_leaf +   
               booster_config->l2_reg) continue;
            
            if(right_matrix[0] < booster_config->min_sum_hessian_in_leaf +
               booster_config->l2_reg) break;
            
            double left_gain = Solve(left_matrix, left_vec, left_k, cur_num_vars + 1, l3_reg, row_size + 1);
            
            double right_gain = Solve(right_matrix, right_vec, right_k, cur_num_vars + 1, l3_reg, row_size + 1);
            
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
        
#pragma omp simd aligned(right_histogram_ptr,sum_up_data:32)
        for(int i = 0; i < row_size; ++i) {
            right_histogram_ptr[i] = sum_up_data[i];
        }
        
        for(int i = 0; i < num_bins - 1; ++i) {
            
            const double *bin_histogram = histograms[i].data();
            
            if(bin_histogram[1] == 0.0) {
                continue;
            }
            
#pragma omp simd aligned(right_histogram_ptr,left_histogram_ptr,bin_histogram:32)
            for(int i = 0; i < row_size; ++i) {
                right_histogram_ptr[i] -= bin_histogram[i];
                left_histogram_ptr[i] += bin_histogram[i];
            }
            
    
            LoadMatrix(left_histogram, prev_num_vars, left_matrix, left_vec);
            LoadMatrix(right_histogram, prev_num_vars, right_matrix, right_vec);
            
            if(left_matrix[0] < booster_config->min_sum_hessian_in_leaf +
               booster_config->l2_reg) continue;
            
            if(right_matrix[0] < booster_config->min_sum_hessian_in_leaf +
               booster_config->l2_reg) break;
            
            double left_gain = Solve(left_matrix, left_vec, left_k,
                                     cur_num_vars + 1, l3_reg, row_size - 1 - prev_num_vars);
            
            double right_gain = Solve(right_matrix, right_vec, right_k,
                                      cur_num_vars + 1, l3_reg, row_size - 1 - prev_num_vars);
            
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

double LinearBinFeature::Solve(const dvec32 &matrix,
                               const dvec32 &vec,   
                               dvec64 &ks,
                               int n, double l3_reg, int row_size) {
    double t_start = omp_get_wtime();
    SolveMKL(matrix, vec, n, row_size, ks);
    double t_end = omp_get_wtime();
    mkl_time += (t_end - t_start);
    const double *vec_ptr = vec.data(), *ks_ptr = ks.data();
    double loss = 0.0;
#pragma omp simd aligned(vec_ptr,ks_ptr:32)
    for(int i = 0; i < n; ++i) {
        loss -= 0.5 * ks_ptr[i] * vec_ptr[i];
        loss += l3_reg * fabs(ks_ptr[i]);
    }
    return loss;
}

void LinearBinFeature::Split(int leaf_id, SplitInfo *split, const vector<uint8_t> &bit_vector,      
                             int num_left, int num_leaf_data, int left_start, int left_end,
                             int right_start, int right_end) {
    /*if(leaf_id == 0) {
        int left_child = split->left_leaf_id;
        int right_child = split->right_leaf_id;
        leaf_local_data[left_child].clear();
        leaf_local_data[left_child].resize(num_left + 8, 0);
        leaf_local_data[right_child].clear();
        leaf_local_data[right_child].resize(num_data - num_left + 8, 0);
        uint8_t *left_local_data = leaf_local_data[left_child].data();
        uint8_t *right_local_data = leaf_local_data[right_child].data();  
        uint8_t *parent_local_data = root_data_ptr->data();
        //WARN: out of bound read from linear_bin_feature.data
        int left_cnt = 0, right_cnt = 0;
        for(int i = 0; i < num_data; i += 8) {
            uint64_t bits = *(uint64_t*)(parent_local_data + i);
            uint64_t masks = *(uint64_t*)(bit_vector.data() + i);
            
            uint64_t left = _pext_u64(bits, masks);
            uint64_t right = _pext_u64(bits, ~masks);
            
            //intel uses little endian, so this should work
            *(uint64_t*)(left_local_data + left_cnt) = left;
            *(uint64_t*)(right_local_data + right_cnt) = right;
            uint64_t to_left = _mm_popcnt_u64(masks);
            to_left = to_left >> 3;
            left_cnt += to_left;
            right_cnt += (8 - to_left);
        }
    }
    else {
        int left_child = split->left_leaf_id;
        int right_child = split->right_leaf_id;
        int leaf_data_size = global_leaf_ends[leaf_id] - global_leaf_starts[leaf_id];           
        //+8 guarantees that the masked vectorization operation will not be out of bound.   
        int packed_num_left = num_left + 8;
        int packed_num_right = (leaf_data_size - num_left) + 8;
        leaf_local_data[left_child].clear();
        leaf_local_data[left_child].resize(packed_num_left, 0);
        leaf_local_data[right_child].clear(); 
        leaf_local_data[right_child].resize(packed_num_right, 0);
        uint8_t *left_local_data = leaf_local_data[left_child].data();
        uint8_t *right_local_data = leaf_local_data[right_child].data();  
        uint8_t *parent_local_data = leaf_local_data[leaf_id].data();
        int left_cnt = 0, right_cnt = 0;
        for(int i = 0; i < leaf_data_size; i += 8) {
            uint64_t bits = *(uint64_t*)(parent_local_data + i);
            uint64_t masks = *(uint64_t*)(bit_vector.data() + i);
            uint64_t left = _pext_u64(bits, masks);
            uint64_t right = _pext_u64(bits, ~masks);
            
            //intel uses little endian, so this should work
            *(uint64_t*)(left_local_data + left_cnt) = left;
            *(uint64_t*)(right_local_data + right_cnt) = right;
            uint64_t to_left = _mm_popcnt_u64(masks);
            to_left = to_left >> 3;
            left_cnt += to_left;
            right_cnt += (8 - to_left);
        }
        leaf_local_data[leaf_id].clear();
        leaf_local_data[leaf_id].shrink_to_fit();   
    }*/
    if(leaf_id == 0) {
        uint8_t *left_local_data = copy_tmp.data();
        uint8_t *right_local_data = copy_tmp.data() + num_left + 8;
        uint8_t *parent_local_data = root_data_ptr->data();//data.data();
        //WARN: out of bound read from linear_bin_feature.data
        int left_cnt = 0, right_cnt = 0;
        for(int i = 0; i < num_data; i += 8) {
            uint64_t bits = *(uint64_t*)(parent_local_data + i);
            uint64_t masks = *(uint64_t*)(bit_vector.data() + i);
            
            uint64_t left = _pext_u64(bits, masks);
            uint64_t right = _pext_u64(bits, ~masks);
            
            //intel uses little endian, so this should work
            *(uint64_t*)(left_local_data + left_cnt) = left;
            *(uint64_t*)(right_local_data + right_cnt) = right;
            uint64_t to_left = _mm_popcnt_u64(masks);
            to_left = to_left >> 3;
            left_cnt += to_left;
            right_cnt += (8 - to_left);
        }
        memcpy(leaf_local_data.data() + left_start, copy_tmp.data(), num_left * sizeof(uint8_t));
        memcpy(leaf_local_data.data() + right_start,
               copy_tmp.data() + num_left + 8,
               (num_leaf_data - num_left) * sizeof(uint8_t));
    }
    else {
        int leaf_data_size = global_leaf_ends[leaf_id] - global_leaf_starts[leaf_id];
        uint8_t *parent_local_data = leaf_local_data.data() + left_start;
        uint8_t *left_local_data = copy_tmp.data();
        uint8_t *right_local_data = copy_tmp.data() + num_left + 8;
        int left_cnt = 0, right_cnt = 0;
        for(int i = 0; i < leaf_data_size; i += 8) {
            uint64_t bits = *(uint64_t*)(parent_local_data + i);
            uint64_t masks = *(uint64_t*)(bit_vector.data() + i);           
            uint64_t left = _pext_u64(bits, masks);
            uint64_t right = _pext_u64(bits, ~masks);
            
            //intel uses little endian, so this should work
            *(uint64_t*)(left_local_data + left_cnt) = left;
            *(uint64_t*)(right_local_data + right_cnt) = right;
            uint64_t to_left = _mm_popcnt_u64(masks);
            to_left = to_left >> 3;
            left_cnt += to_left;
            right_cnt += (8 - to_left);
        }
        memcpy(leaf_local_data.data() + left_start, copy_tmp.data(), num_left * sizeof(uint8_t));
        memcpy(leaf_local_data.data() + right_start,
               copy_tmp.data() + num_left + 8,
               (leaf_data_size - num_left) * sizeof(uint8_t));
    }
}

const uint8_t* LinearBinFeature::GetLocalData(int leaf_id, int leaf_start) {
    if(leaf_id == 0) {
        return root_data_ptr->data();
    }
    else {
        return leaf_local_data.data() + leaf_start;
    }
}


void LinearBinFeature::GetLeafSumUp(dvec64& leaf_sum_up) {
    int row_sizze = 2 + 3 * prev_num_vars + prev_num_vars * (prev_num_vars - 1) / 2;
    leaf_sum_up.clear();
    leaf_sum_up.resize(row_sizze, 0.0);
    vector<vector<double>> thread_leaf_sum_up(booster_config->num_threads);
#pragma omp parallel for schedule(static, 1) num_threads(booster_config->num_threads)
    for(int i = 0; i < booster_config->num_threads; ++i) {
        thread_leaf_sum_up[i].clear();
        thread_leaf_sum_up[i].resize(row_sizze, 0.0);
    }
    
#pragma omp parallel for schedule(static) num_threads(booster_config->num_threads)
    for(int i = 0; i < num_bins; ++i) {
        const dvec32& bin_histogrma = (*tmp_histograms)[i];
        int omp_thread_num = omp_get_thread_num();
        vector<double> &sum_up = thread_leaf_sum_up[omp_thread_num];
#pragma omp simd
        for(int j = 0; j < row_sizze; ++j) {
            sum_up[j] += bin_histogrma[j];
        }
    }
    
    for(int i = 0; i < row_sizze; ++i) {
        for(int j = 0; j < booster_config->num_threads; ++j) {
            leaf_sum_up[i] += thread_leaf_sum_up[j][i];                 
        }
    }   
}

void LinearBinFeature::SolveMKL(const dvec32 &matrix,
                                const dvec32 &vec,
                                int n, int row_size, dvec64 &ks) {
    const double *matrix_ptr = matrix.data(), *vec_ptr = vec.data();
    double *ks_ptr = ks.data();
    double *matrix_copy_ptr = matrix_copy;
    int cnt = 0;
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < i; ++j) {
            matrix_copy_ptr[j * n + i] = matrix_copy_ptr[i * n + j] = matrix_ptr[cnt++];
        }
        matrix_copy_ptr[i * n + i] = matrix_ptr[cnt++];
    }
#pragma omp simd aligned(vec_ptr,ks_ptr:32)
    for(int i = 0; i < n; ++i) {
        ks_ptr[i] = vec_ptr[i];
    }
    
    //dppsv("U", &n, &mkl_m, matrix_copy, ks.data(), &n, &mkl_info);
    dgesv(&n, &mkl_m, matrix_copy, &n, mkl_ipiv, ks_ptr, &n, &mkl_info);
}

void LinearBinFeature::LoadMatrix(const dvec32 &histogram,
                                  int num_vars, dvec32 &matrix,
                                  dvec32 &vec) {
    double* matrix_ptr = matrix.data();
    const double* histogram_ptr = histogram.data();
    double reg = booster_config->l2_reg;
    matrix_ptr[0] = histogram_ptr[1] + reg;
    vec[0] = -histogram_ptr[0];
    histogram_ptr += 2;
    matrix_ptr += 1;
    for(int i = 0; i < num_vars; ++i) {
        matrix_ptr[0] = histogram_ptr[1];
#pragma omp simd
        for(int j = 1; j < i + 1; ++j) {
            matrix_ptr[j] = histogram_ptr[2 + j];
        }
        matrix_ptr[i + 1] = histogram_ptr[2] + reg; 
        vec[i + 1] = -histogram_ptr[0];
        matrix_ptr += (i + 2);
        histogram_ptr += (i + 3);
    }
}

int LinearBinFeature::SplitIndex(uint8_t threshold, int leaf_id, int split_start, int split_end,
                                 uint8_t *bit_vector,
                                 int *data_indices, int *left_data_indices,
                                 int *right_data_indices, int leaf_start) {
    const uint8_t *leaf_data_bins = GetLocalData(leaf_id, leaf_start) + split_start;    
    
    int left_cnt = 0;
    int right_cnt = 0;
    for(int i = 0; i < split_end - split_start; ++i) {
        if(leaf_data_bins[i] <= threshold) {
            bit_vector[i] = 0xff;
            left_data_indices[left_cnt++] = data_indices[i];        
        }
        else {
            bit_vector[i] = 0x00;
            right_data_indices[right_cnt++] = data_indices[i];  
        }
    }
    return left_cnt; 
}

void LinearBinFeature::BeforeTrain() {  
    /*for(int i = 0; i < num_bins; ++i) {
        bin_values[i] /= bin_counts[i];
    }*/
}

/*void LinearBinFeature::SumUpLeafPredictValues(int leaf_id, const int *data_indices,
                                              int leaf_num_data, double k,
                                              double *predict_values) {
    const uint8_t *local_bin_data = leaf_local_data.data() + global_leaf_starts[leaf_id];
    for(int i = 0; i < leaf_num_data; ++i) {
        int index = data_indices[i];
        predict_values[index] += k * bin_values[local_bin_data[i]];
    }
}*/ 

/*function<double()> LinearBinFeature::GetLeafDataIterator(int leaf_id) {
    vector<double> &bin_valuess = bin_values;
    vector<uint8_t> &leaf_data = leaf_local_data[leaf_id];          
    int cur_pos = 0;
    return ([this, &bin_valuess, &leaf_data, cur_pos]
            () mutable -> double {
        return bin_valuess[leaf_data[cur_pos++]];
    });
}*/

void LinearBinFeature::Sample(int *sampled_data_indices, int sampled_num_data) {
    num_data = sampled_num_data;
    //leaf_local_data[0].resize(num_data + 8, 0);
    root_data_ptr = &leaf_local_data;
    for(int i = 0; i < num_data; ++i) {
        leaf_local_data[i] = data[sampled_data_indices[i]];
    }   
}
