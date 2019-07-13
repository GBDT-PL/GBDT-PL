//
//  datamat.cpp
//  LinearGBM
//


//

#include "datamat.hpp"
#include <cassert>
#include <fstream>
#include <limits>
#include <omp.h>

using std::vector;
using std::string;
using std::ifstream;
using std::cout;
using std::endl;

DataMat::DataMat(const BoosterConfig& _booster_config,
                 string _name, int _label_idx, int _query_idx,
                 string fname,
                 DataMat* _reference):
booster_config(_booster_config), name(_name) {
    label_idx = _label_idx;
    query_idx = _query_idx;
    reference = _reference;
    
    if(reference != nullptr) {
        csv_fname = fname;
        DataReader reader(fname, 50000, unbined_data, label, label_idx);
        reader.ReadByRow(booster_config.num_threads);
        num_feature = reader.get_num_features();
        num_data = reader.get_num_data();
        CalcNormStatsAndNormalize(unbined_data, reference);
	    cout << "[GBDT-PL] finish loading " << name << ": " << num_data << " data points with " << num_feature << " features." << endl; 
    }
    else {
        vector<vector<double>> feature_matrix;
        csv_fname = fname;
        DataReader reader(fname, 50000, feature_matrix, label, label_idx);

        reader.Read(booster_config.num_threads); 
        num_feature = reader.get_num_features();
        num_data = reader.get_num_data();

        CalcNormStatsAndNormalize(feature_matrix, reference);
        
        sparse_ratio.resize(num_feature, 0.0);
        
        CalcBinBoundaries(feature_matrix);
        
        feature_matrix.clear();
        feature_matrix.shrink_to_fit(); 
        cout << "[GBDT-PL] finish loading " << name << ": " << num_data << " data points with " << num_feature << " features." << endl;  
    }
}

void DataMat::CalcNormStatsAndNormalize(vector<vector<double>>& feature_matrix, DataMat* reference) {
    if(reference == nullptr) {
        norm_bias.resize(num_feature);
        norm_scale.resize(num_feature);
        if(booster_config.normalization == "min_max") {
            #pragma omp parallel for schedule(static) num_threads(booster_config.num_threads)
            for(int fid = 0; fid < num_feature; ++fid) {
                double feature_max = std::numeric_limits<double>::min();
                double feature_min = std::numeric_limits<double>::max();
                for(int idx = 0; idx < num_data; ++idx) {
                    double fval = feature_matrix[fid][idx];
                    if(fval > feature_max) {
                        feature_max = fval;
                    }
                    if(fval < feature_min) {
                        feature_min = fval;
                    }
                }
                if(feature_max == feature_min) {
                    norm_scale[fid] = 1.0;
                    norm_bias[fid] = feature_min;
                }
                else {
                    norm_scale[fid] = feature_max - feature_min;
                    norm_bias[fid] = feature_min;
                }
            }
        }
        else if(booster_config.normalization == "mean_std") {
            #pragma omp parallel for schedule(static) num_threads(booster_config.num_threads)
            for(int fid = 0; fid < num_feature; ++fid) {
                double feature_sum = 0.0, feature_sum_square = 0.0;
                for(int idx = 0; idx < num_data; ++idx) {
                    double fval = feature_matrix[fid][idx];
                    feature_sum += fval;
                    feature_sum_square += fval * fval;
                }
                double mean = feature_sum / num_data;
                double stdev = (feature_sum_square / num_data) - mean * mean;
                if(stdev <= 1e-15) {
                    norm_bias[fid] = mean;
                    norm_scale[fid] = 1.0;
                }
                else {
                    norm_bias[fid] = mean;
                    norm_scale[fid] = stdev;
                }
            }
        }
        if(booster_config.normalization != "no") {
            #pragma omp parallel for schedule(static) num_threads(booster_config.num_threads)
            for(int fid = 0; fid < num_feature; ++fid) {
                double bias = norm_bias[fid];
                double scale = norm_scale[fid];
                for(int idx = 0; idx < num_data; ++idx) {
                    double fval = feature_matrix[fid][idx];
                    feature_matrix[fid][idx] = (fval - bias) / scale;
                }
            }
        }
    }
    else {
        norm_bias = reference->norm_bias;
        norm_scale = reference->norm_scale;
        for(int fid = 0; fid < num_feature; ++fid) {
            double bias = norm_bias[fid];
            double scale = norm_scale[fid];
            cout << "test feature " << fid << " bias " << bias << " sacle " << scale << endl;  
        }
        cout << "num_feature " << num_feature << " num_data " << num_data << endl;  
        if(booster_config.normalization != "no") {
            #pragma omp parallel for schedule(static) num_threads(booster_config.num_threads)
            for(int fid = 0; fid < num_feature; ++fid) {
                double bias = norm_bias[fid];
                double scale = norm_scale[fid];
                for(int idx = 0; idx < num_data; ++idx) {
                    double fval = unbined_data[idx][fid];
                    unbined_data[idx][fid] = (fval - bias) / scale;
                }
            }
        }
    }
}

void DataMat::CalcBinBoundaries(vector<vector<double> > &feature_matrix) {
    bin_boundaries.clear();
    bin_boundaries.resize(num_feature);
    num_bins_per_feature.resize(num_feature, 0);
    bin_data.resize(num_feature);
    bin_counts.resize(num_feature);
    bin_values.resize(num_feature);
#pragma omp parallel for schedule(static) num_threads(booster_config.num_threads)
    for(int i = 0; i < feature_matrix.size(); ++i) {
        vector<int> indices(num_data, -1);
#pragma omp simd
        for(int j = 0; j < num_data; ++j) {
            indices[j] = j;
        }
        bin_data[i].resize(num_data, 0);
        bin_counts[i].resize(booster_config.max_bin, 0);
        bin_values[i].resize(booster_config.max_bin, 0.0);
        std::sort(indices.begin(), indices.end(), [&feature_matrix, i] (int a, int b)   
                  { return feature_matrix[i][a] < feature_matrix[i][b]; } );
        std::sort(feature_matrix[i].begin(), feature_matrix[i].end());
        num_bins_per_feature[i] = CalcBinBoundary(feature_matrix[i], bin_boundaries[i],
                                                  indices, bin_data[i], bin_counts[i], bin_values[i]); 
    }
}

int DataMat::CalcBinBoundary(vector<double> &sorted_feature, vector<double>& bin_boundary,
                              vector<int> &data_indices, vector<uint8_t> &bin_data,
                             vector<int>& bin_counts, vector<double>& bin_values) {
    vector<double> unique_values;
    vector<int> value_count;
    int num_values = 0;
    
    unique_values.push_back(sorted_feature[0]); 
    value_count.push_back(1);
    num_values += 1;
    
    for(int i = 1; i < sorted_feature.size(); ++i) {
        if(sorted_feature[i] != sorted_feature[i - 1]) {
            unique_values.push_back(sorted_feature[i]); 
            value_count.push_back(1);
            num_values += 1;
        }
        else {
            ++value_count.back();
        }
    }
    
    int sum_of_count = 0;
    for(int i = 0; i < num_values; ++i) {
        sum_of_count += value_count[i];
    }
    
    if(num_values <= booster_config.max_bin) {
        int acc_bin_cnt = 0;
        for(int i = 0; i < num_values - 1; ++i) {
            
            for(int j = acc_bin_cnt; j < acc_bin_cnt + value_count[i]; ++j) {
                bin_data[data_indices[j]] = static_cast<int>(i);
                bin_values[i] += sorted_feature[j];
            }
            acc_bin_cnt += value_count[i];
            bin_counts[i] = value_count[i];
            
            if(unique_values[i + 1] == 0.0) {
                bin_boundary.push_back(-1e-10);     
            }
            else if(unique_values[i] == 0.0) {
                bin_boundary.push_back(1e-10);
            }
            else {
                bin_boundary.push_back((unique_values[i] + unique_values[i + 1]) / 2.0);
            }
        }
        for(int j = acc_bin_cnt; j < num_data; ++j) {
            bin_data[data_indices[j]] = static_cast<uint8_t>(num_values - 1);
            bin_values[num_values - 1] += sorted_feature[j];
        }
        bin_counts[num_values - 1] = value_count[num_values - 1];
        
        for(int i = 0; i < num_values; ++i) {
            bin_values[i] /= bin_counts[i];
        }
	    bin_values.resize(num_values);
        bin_values.shrink_to_fit();
        bin_counts.resize(num_values);
        bin_counts.shrink_to_fit();
        return num_values;
    }
    else {
        int rest_bin = booster_config.max_bin;
        int rest_count = static_cast<int>(sorted_feature.size());// + num_zero;
        int mean_count_per_bin = static_cast<int>(sorted_feature.size()) / booster_config.max_bin;
        vector<bool> is_big_bin(num_values, false);
        for(int i = 0; i < num_values; ++i) {
            if(value_count[i] > mean_count_per_bin || unique_values[i] == 0.0) {
                --rest_bin;
                rest_count -= value_count[i];
                is_big_bin[i] = true;
            }
        }
        mean_count_per_bin = rest_count / rest_bin; 
        int cur_bin_cnt = 0;
        int acc_bin_cnt = 0;
        int bin_cnt = 0;
        for(int i = 0; i < num_values - 1; ++i) {
            cur_bin_cnt += value_count[i];
            if(!is_big_bin[i]) {
                rest_count -= value_count[i];
            }
            if(is_big_bin[i] || cur_bin_cnt >= mean_count_per_bin ||
               (is_big_bin[i + 1] && cur_bin_cnt >= std::fmax(1, 0.5 * mean_count_per_bin))  ||
               unique_values[i] == 0.0 || unique_values[i + 1] == 0.0) {
                for(int j = acc_bin_cnt; j < acc_bin_cnt + cur_bin_cnt; ++j) {
                    bin_data[data_indices[j]] = static_cast<uint8_t>(bin_cnt);
                    bin_values[bin_cnt] += sorted_feature[j];
                }
                bin_counts[bin_cnt] = cur_bin_cnt;
                ++bin_cnt;
                acc_bin_cnt += cur_bin_cnt;
                cur_bin_cnt = 0;
                if(unique_values[i + 1] == 0.0) {
                    bin_boundary.push_back(-1e-10);
                }
                else if(unique_values[i] == 0.0) {
                    bin_boundary.push_back(1e-10);  
                }
                else {
                    bin_boundary.push_back((unique_values[i] + unique_values[i + 1]) / 2.0);    
                }
                if(bin_cnt >= booster_config.max_bin - 1) {
                    break;
                }
                if(!is_big_bin[i] && rest_bin > 1) {    
                    rest_bin -= 1;
                    mean_count_per_bin = rest_count / rest_bin; 
                }
            }
        }
        for(int j = acc_bin_cnt; j < num_data; ++j) {
            bin_data[data_indices[j]] = static_cast<uint8_t>(bin_cnt);
            bin_values[bin_cnt] += sorted_feature[j];
        }
        bin_counts[bin_cnt] = num_data - acc_bin_cnt; 
        ++bin_cnt;
        for(int i = 0; i < bin_cnt; ++i) {
            bin_values[i] /= bin_counts[i];
        }
	    bin_values.resize(bin_cnt);
        bin_values.shrink_to_fit();
        bin_counts.resize(bin_cnt);
        bin_counts.shrink_to_fit();
        return bin_cnt;
    }
}

void DataMat::Shrink() {
    if(reference == nullptr) {
        unbined_data.clear();
        unbined_data.shrink_to_fit();
    }
}
