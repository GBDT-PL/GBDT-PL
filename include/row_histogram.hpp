//
//  row_histogram.hpp
//  LinearGBMVector
//


//

#ifndef row_histogram_hpp
#define row_histogram_hpp

#include <stdio.h>
#include <vector>
#include <map>
#include <string>
#include <cassert>
#include <iostream>
#include "alignment_allocator.hpp"

using std::vector;
using std::string;
using std::map;

class RowHistogram {
private:
    vector<dvec32> *sub_histograms;
public:
    int max_bin;
    int leaf_id;
    int feature_id;
    int depth;
    int max_var;
    int row_size;
    int cur_var;
    bool need_augment;
    
    RowHistogram(int _max_bin, int _feature_id, int _max_var) {
        max_bin = _max_bin;
        feature_id = _feature_id;   
        sub_histograms = new vector<dvec32>(max_bin);
        
        max_var = _max_var;
        
        int max_row_size = 2 + 3 * max_var;
        if(max_var >= 2) {
            max_row_size += max_var * (max_var - 1) / 2;    
        }
        for(int i = 0; i < max_bin; ++i) {
            //+8 to avoid write out of bound in linear_bin_feature histogram construction
            (*sub_histograms)[i].resize(max_row_size + 8, 0.0);
        }
    }
    
    void SetDepthAndLeafID(int _cur_var, int _leaf_id, bool _need_augment) {
        cur_var = _cur_var; 
        row_size = 2 + 3 * cur_var;
        if(cur_var >= 2) {
            row_size += cur_var * (cur_var - 1) / 2;
        }
        need_augment = _need_augment; 
    }
    
    vector<dvec32>* Get() {
        return sub_histograms;          
    }
    
    void Clear() {
        for(int i = 0; i < max_bin; ++i) {
            dvec32& histogram = (*sub_histograms)[i];
#pragma omp simd
            for(int j = 0; j < row_size; ++j) {
                histogram[j] = 0.0;
            }
        }
    }
    
    void Substract(RowHistogram* child);    
};

#endif /* row_histogram_hpp */
