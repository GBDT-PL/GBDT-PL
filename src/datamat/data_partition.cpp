//
//  data_partition.cpp
//  LinearGBM
//


//

#include "data_partition.hpp"
#include <omp.h>
#include <cstring>
#include <string>
#include <cmath>
#include <set>
#include <fstream>  
#include <cstdlib>
#include <ctime>
#include <x86intrin.h>

using std::vector;

DataPartition::DataPartition(DataMat* _train_set, BoosterConfig* _booster_config, fvec64 &gradientss):
all_gradients(gradientss) {
    train_set = _train_set;
    booster_config = _booster_config;
    data_indices.resize(_train_set->num_data, -1);
    data_indices_in_left.resize(_train_set->num_data, -1);
    data_indices_in_right.resize(_train_set->num_data, -1);
    leaf_starts.resize(2 * _booster_config->max_leaf + 1, -1);      
    leaf_ends.resize(2 * _booster_config->max_leaf + 1, -1);

    pred_values.resize(train_set->num_data, 0.0);   
    
    leaf_starts[0] = 0;
    leaf_ends[0] = train_set->num_data;
    num_data = train_set->num_data;
    if(train_set->num_data % 4 == 0) {
        num_data_aligned = train_set->num_data;
    }
    else {
        num_data_aligned = (train_set->num_data / 4 + 1) * 4;
    }

    bit_vector.resize(train_set->num_data + 8, 0x00);
    
#pragma omp parallel for schedule(static) num_threads(booster_config->num_threads)
    for(int i = 0; i < train_set->num_data; ++i) {
        data_indices[i] = i;
    }

    histogram_repo.resize(booster_config->max_leaf, nullptr);
    for(int i = 0; i < booster_config->max_leaf; ++i) {
        histogram_repo[i] = new vector<RowHistogram*>(train_set->num_feature);
    }
    
    
    bin_gradients.resize(5 * train_set->num_data, 0.0); 
    
    features.resize(train_set->num_feature, nullptr);                   
    
    left_copy_offsets.resize(booster_config->num_threads + 1, 0);
    right_copy_offsets.resize(booster_config->num_threads + 1, 0);

    int least_bin = booster_config->max_bin;
    least_bin_feature = 0;
    for(int i = 0; i < train_set->num_feature; ++i) {
        if(train_set->num_bins_per_feature[i] > 1) {
            useful_features.push_back(i);
            if(train_set->num_bins_per_feature[i] < least_bin) {
                least_bin = train_set->num_bins_per_feature[i];     
                least_bin_feature = i;
            }
        }
    }

#pragma omp parallel for schedule(static) num_threads(booster_config->num_threads)
    for(int i = 0; i < train_set->num_feature; ++i) {
        /*if(train_set->sparse_ratio[i] <= booster_config->sparse_threshold && i != least_bin_feature) {
            features[i] = new SparseLinearBinFeature(train_set, false, i, booster_config,
                                                     leaf_starts, leaf_ends, booster_config->num_vars,
                                                     train_set->get_bin_data(i),
                                                     train_set->get_bin_count(i),
                                                     train_set->get_bin_value(i)) ;
        }
        else {*/
            features[i] = new LinearBinFeature(train_set, false, i, booster_config,
                                           leaf_starts, leaf_ends, booster_config->num_vars,    
                                               train_set->get_bin_data(i),
                                               train_set->get_bin_count(i),
                                               train_set->get_bin_value(i)); 
        //}
    } 

    //PushDataIntoFeatures();
    
    int max_var = booster_config->num_vars;
    if(booster_config->leaf_type == "additive_linear") {
        max_var = 2;
    }
    for(int i = 0; i < booster_config->max_leaf; ++i) {
        for(int j = 0; j < train_set->num_feature; ++j) {
            (*histogram_repo[i])[j] = new RowHistogram(features[j]->GetNumBins(),
                                                       j, max_var); 
        }
    }
    
    first_unused_histogram = 0;
    
    train_set->weights.resize(train_set->num_data, 1.0);
}

void DataPartition::BeforeTrainTree(int cur_class, int iteration) {
    
#pragma omp parallel for schedule(static) num_threads(booster_config->num_threads)
    for(int i = 0; i < train_set->num_feature; ++i) {
        features[i]->BeforeTrainTree(iteration, cur_class, data_indices, leaf_starts, leaf_ends); 
    }
    
#pragma omp parallel for schedule(static) num_threads(booster_config->num_threads)
    for(int i = 0; i < train_set->num_data; ++i) {
        data_indices[i] = i;
    }
    
    first_unused_histogram = 0;
}

void DataPartition::BeforeTrain() {
    int small_features = 0;
    for(int i = 0; i < train_set->num_feature; ++i) {
        int max_bin_count = 0;
        int max_count_bin = -1;
        if(features[i]->GetNumBins() <= 10) {
            ++small_features; 
        }
        for(int j = 0; j < features[i]->GetNumBins(); ++j) {
            if((static_cast<LinearBinFeature*>(features[i]))->bin_counts[j] > max_bin_count) {
                max_bin_count = (static_cast<LinearBinFeature*>(features[i]))->bin_counts[j];       
                max_count_bin = j;
            }
        }
        cout << "[GBDT-PL] feature " << i << " max_bin_count " << max_bin_count << " max_count_bin " << max_count_bin << endl;
    }
    
#pragma omp parallel for schedule(guided) num_threads(booster_config->num_threads)
    for(int i = 0; i < train_set->num_feature; ++i) {
        if(features[i]->useful) {
            features[i]->BeforeTrain();
        }
    }
    
    std::cout << "[GBDT-PL] useful features " << useful_features.size() << std::endl;
    std::cout << "[GBDT-PL] small features " << small_features << std::endl << std::endl;
}

void DataPartition::LeafSumUp() {
    features[least_bin_feature]->GetLeafSumUp(leaf_sum_up);
}

void DataPartition::Split(int leaf_id, SplitInfo* split, int cur_class) {
    int num_threads = booster_config->num_threads;                  
    
    int start = leaf_starts[leaf_id], end = leaf_ends[leaf_id]; 
    
    uint8_t threshold = static_cast<uint8_t>(split->threshold);
    int *local_data_indices = data_indices.data() + start;
    int data_size = end - start;
    int chunk_size = (data_size + num_threads - 1) / num_threads;
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for(int i = 0; i < num_threads; ++i) {
        int split_start = chunk_size * i;
        int split_end = std::min(split_start + chunk_size, data_size);  
        if(split_end < split_start) {
            split_start = split_end;
        }
        left_copy_offsets[i + 1] = features[split->feature_index]->SplitIndex(threshold, leaf_id, split_start, split_end, bit_vector.data() + split_start,
            local_data_indices + split_start,
            data_indices_in_left.data() + split_start,
            data_indices_in_right.data() + split_start, start);
        right_copy_offsets[i + 1] = split_end - split_start - left_copy_offsets[i + 1];
    }
    
    for(int i = 0; i < num_threads; ++i) {
        left_copy_offsets[i + 1] += left_copy_offsets[i];
        right_copy_offsets[i + 1] += right_copy_offsets[i];
    }
    
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for(int i = 0; i < num_threads; ++i) {
        std::memcpy(data_indices.data() + left_copy_offsets[i] + start, 
                    data_indices_in_left.data() + i * chunk_size,
                    (left_copy_offsets[i + 1] - left_copy_offsets[i]) * sizeof(int));
        std::memcpy(data_indices.data() + left_copy_offsets[num_threads] + right_copy_offsets[i] + start,
                    data_indices_in_right.data() + i * chunk_size,
                    (right_copy_offsets[i + 1] - right_copy_offsets[i]) * sizeof(int)); 
    }
    
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for(int i = 0; i < useful_features.size(); ++i) {
        int fid = useful_features[i];   
        features[fid]->Split(leaf_id, split, bit_vector,
                             left_copy_offsets[num_threads], end - start,   
                             start, start + left_copy_offsets[num_threads],
                             start + left_copy_offsets[num_threads], end);
    }
    
    leaf_starts[split->left_leaf_id] = start;
    leaf_starts[split->right_leaf_id] = start + left_copy_offsets[num_threads];
    leaf_ends[split->left_leaf_id] = start + left_copy_offsets[num_threads];
    leaf_ends[split->right_leaf_id] = end;
}

vector<RowHistogram*>* DataPartition::GetHistogramFromPool() {  
    assert(first_unused_histogram < booster_config->max_leaf);      
    return histogram_repo[first_unused_histogram++];    
}

void DataPartition::PushDataIntoFeatures() {    
    std::ifstream fin(train_set->csv_fname);
    double fvalue = 0.0;
    int feature_idx = 0, data_idx = 0;
    char delimiter = '\0';
    while(fin >> fvalue) {
        if(feature_idx != train_set->label_idx && feature_idx != train_set->query_idx) {    
            int true_feature_idx = train_set->feature_idx_map[feature_idx];
            assert(true_feature_idx != -1 || fvalue == 0.0);
            if(true_feature_idx != -1) {
                (features[true_feature_idx])->PushIntoBin(fvalue, data_idx);
            }
        }
        delimiter = fin.get();
        if(delimiter == ',') {
            feature_idx += 1;
        }
        else if(delimiter == '\r' || delimiter == '\n') {
            feature_idx = 0;                                
            data_idx += 1;  
        }
    }
}

void DataPartition::PrintAllTime() {
    booster_config->PrintAllTime();
    double mkl_time = 0.0;
    double load_matrix_time = 0.0;
    for(int i = 0; i < train_set->num_feature; ++i) {
        mkl_time += features[i]->mkl_time;
        load_matrix_time += features[i]->load_matrix_time;
    }
    cout << "mkl_time " << mkl_time << endl;
    cout << "load_matrix_time " << load_matrix_time << endl;
}

void DataPartition::AfterTrainTree() {
#pragma omp parallel for schedule(static) num_threads(booster_config->num_threads)
    for(int i = 0; i < train_set->num_feature; ++i) {
        features[i]->AfterTrainTree(); 
    }
}

void DataPartition::Sample(float *gradients_ptrr, int num_dataa) {      
    num_data = num_dataa;
    
    leaf_starts[0] = 0;
    leaf_ends[0] = num_data;
#pragma omp parallel for schedule(static) num_threads(booster_config->num_threads)
    for(int i = 0; i < train_set->num_feature; ++i) {
        features[i]->Sample(data_indices.data(), num_data); 
    }
}
