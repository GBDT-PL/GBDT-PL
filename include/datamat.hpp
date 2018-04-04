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

using std::vector;
using std::string;

class DataMat {
private:
    const BoosterConfig& booster_config;
    
    void CalcBinBoundaries(vector<vector<double>> &feature_matrix);
    
    int CalcBinBoundary(vector<double> &sorted_feature, vector<double> & bin_boundary); 
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
    vector<vector<double>> data;
    vector<vector<double>> unbined_data;
    vector<vector<double>> bin_boundaries;  
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
};

#endif /* datamat_hpp */
