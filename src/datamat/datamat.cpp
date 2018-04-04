//
//  datamat.cpp
//  LinearGBM
//


//

#include "datamat.hpp"
#include <cassert>
#include <fstream>
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
        /*csv_fname = fname;
        ifstream fin(csv_fname);
        double fvalue;
        label.clear();
        queries.clear();
        int data_idx = 0;
        int feature_idx = 0;
        char delimiter = '\0';
        num_feature = reference->num_feature;   
        unbined_data.clear();
        while(fin >> fvalue) {
            if(data_idx >= unbined_data.size()) {
                unbined_data.emplace_back(reference->num_feature, 0.0);
            }
            
            if(feature_idx == label_idx) {
                label.push_back(fvalue);
            }
            else if(feature_idx == query_idx) {
                queries.push_back(fvalue);          
            }
            else {
                int true_feature_idx = reference->feature_idx_map[feature_idx];
                if(true_feature_idx != -1) {
                    unbined_data[data_idx][true_feature_idx] = fvalue;
                }
            }
            delimiter = fin.get();
            if(delimiter == ',') {
                ++feature_idx;
            }
            else if(delimiter == '\n' || delimiter == '\r') {
                feature_idx = 0;
                ++data_idx; 
            }
        }
        num_data = data_idx; */
	csv_fname = fname;
        DataReader reader(fname, 50000, unbined_data, label);
        reader.ReadByRow(booster_config.num_threads);
        num_feature = reader.get_num_features();
        num_data = reader.get_num_data();
    }
    else {
        vector<vector<double>> feature_matrix;
        csv_fname = fname;
        DataReader reader(fname, 50000, feature_matrix, label);
        reader.Read(booster_config.num_threads); 
        num_feature = reader.get_num_features();
        num_data = reader.get_num_data();
        
        feature_idx_map.resize(num_feature + 1, -1);    
        for(int i = 1; i < num_feature + 1; ++i) {
            feature_idx_map[i] = i - 1;
        }
        sparse_ratio.resize(num_feature, 0.0);
        
            /*csv_fname = fname;
            ifstream fin(fname);
            double fvalue;
            label.clear();
            queries.clear();
            vector<vector<double>> feature_matrix;
            int data_idx = 0;
            int feature_idx = 0;
            char delimiter = '\0';
            int cur_feature = 0;
            while(fin >> fvalue) {
                if(feature_idx == label_idx) {
                    label.push_back(fvalue);
                    if(feature_idx_map.size() <= feature_idx) {
                        int cur_size = static_cast<int>(feature_idx_map.size());
                        for(int i = cur_size; i < feature_idx; ++i) {
                            feature_idx_map.push_back(-1);
                        }
                        feature_idx_map.push_back(-1);
                    }
                }
                else if(feature_idx == query_idx) {
                    queries.push_back(static_cast<int>(fvalue));
                    if(feature_idx_map.size() <= feature_idx) {
                        int cur_size = static_cast<int>(feature_idx_map.size());
                        for(int i = cur_size; i < feature_idx; ++i) {
                            feature_idx_map.push_back(-1);
                        }
                        feature_idx_map.push_back(-1);
                    }
                }
                else if(std::fabs(fvalue) >= 1e-15) {
                    if(feature_idx_map.size() <= feature_idx) {
                        int cur_size = static_cast<int>(feature_idx_map.size());
                        for(int i = cur_size; i < feature_idx; ++i) {
                            feature_idx_map.push_back(-1);
                        }
                        feature_idx_map.push_back(cur_feature);
                        ++cur_feature;
                        feature_matrix.emplace_back(0);
                        sparse_ratio.emplace_back(0.0);
                    }
                    else if(feature_idx_map[feature_idx] == -1) {
                        feature_idx_map[feature_idx] = cur_feature;
                        ++cur_feature;
                        feature_matrix.emplace_back(0);
                        sparse_ratio.emplace_back(0.0);
                    }
                    sparse_ratio[feature_idx_map[feature_idx]] += 1.0;
                    feature_matrix[feature_idx_map[feature_idx]].push_back(fvalue); 
                }
                
                delimiter = fin.get();
                if(delimiter == '\r' || delimiter == '\n') {
                    ++data_idx;
                    feature_idx = 0;
                }
                else if(delimiter == ',') {
                    ++feature_idx;
                }
            }
            
            fin.close();
            
            num_feature = cur_feature;
            num_data = data_idx;*/
            CalcBinBoundaries(feature_matrix);
        
        feature_matrix.clear();
        feature_matrix.shrink_to_fit();
        
        /*for(int i = 0; i < sparse_ratio.size(); ++i) {
            sparse_ratio[i] = /= num_data;
        }*/
        
        cout << "finish constructing bin mappers, " << num_data << " and " << num_feature << endl;
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
        //bin_boundaries.emplace_back(0);
        num_bins_per_feature[i] = CalcBinBoundary(feature_matrix[i], bin_boundaries[i],
                                                  indices, bin_data[i], bin_counts[i], bin_values[i]); 
    }
}

int DataMat::CalcBinBoundary(vector<double> &sorted_feature, vector<double>& bin_boundary,
                              vector<int> &data_indices, vector<uint8_t> &bin_data,
                             vector<int>& bin_counts, vector<double>& bin_values) {
    //int num_zero = num_data - static_cast<int>(sorted_feature.size());
    vector<double> unique_values;
    vector<int> value_count;
    int num_values = 0;
    
    /*if(sorted_feature[0] > 0.0 && num_zero > 0) {
        unique_values.push_back(0.0);
        value_count.push_back(num_zero);    
        num_values += 1;
    }*/
    
    unique_values.push_back(sorted_feature[0]); 
    value_count.push_back(1);
    num_values += 1;
    
    for(int i = 1; i < sorted_feature.size(); ++i) {
        if(sorted_feature[i] != sorted_feature[i - 1]) {
            /*if(sorted_feature[i - 1] < 0.0 && sorted_feature[i] > 0.0 && num_zero > 0) {
                unique_values.push_back(0.0);
                value_count.push_back(num_zero);
                num_values += 1;
            }*/
            unique_values.push_back(sorted_feature[i]); 
            value_count.push_back(1);
            num_values += 1;
        }
        else {
            ++value_count.back();
        }
    }
    
    /*if(sorted_feature.back() < 0.0 && num_zero > 0) {
        unique_values.push_back(0.0);
        value_count.push_back(num_zero);
        num_values += 1;
    }*/
    
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
