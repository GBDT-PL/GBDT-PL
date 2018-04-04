//
//  datamat.hpp
//  LinearGBM
//


//

#ifndef datamat_hpp
#define datamat_hpp

#include <vector>
#include "booster_config.hpp"
#include <cmath>
#include <string>
#include <algorithm>
#include "data_reader.hpp" 

using std::vector;
using std::string;

class DataMat {
private:
    const BoosterConfig& booster_config;
    
    void CalcBinBoundaries(vector<vector<double>> &feature_matrix);
    
    int CalcBinBoundary(vector<double> &sorted_feature, vector<double> & bin_boundary,
                        vector<int> &data_indices, vector<uint8_t> &bin_data,
                        vector<int>& bin_counts, vector<double>& bin_values);
public:
    vector<double> weights;
    
    DataMat(const BoosterConfig& _booster_config, 
            string _name,
            int _label_idx,
            int _query_idx,
            string fname,
            DataMat* _reference = nullptr);
    
    void Shrink(); 
    
    vector<int> feature_idx_map;
    vector<vector<int>> bin_counts;
    vector<double> predict_values;
    vector<vector<double>> bin_values; 
    vector<vector<double>> unbined_data;
    vector<vector<double>> bin_boundaries;
    vector<vector<uint8_t>> bin_data;
    int label_idx;
    int query_idx; 
    vector<int> num_bins_per_feature;
    vector<double> label;   
    string name;
    int num_data;
    int num_feature;
    vector<int> queries;
    string csv_fname;
    DataMat* reference;
    vector<double> sparse_ratio;
    
    vector<uint8_t>& get_bin_data(int feature_index) { return bin_data[feature_index]; }
    
    vector<int>& get_bin_count(int feature_index) { return bin_counts[feature_index]; }
    
    vector<double>& get_bin_value(int feature_index) { return bin_values[feature_index]; } 
};

#endif /* datamat_hpp */
