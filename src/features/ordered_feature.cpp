//
//  ordered_feature.cpp
//  LinearGBM
//


//

#include "feature.hpp"
#include <cstring>
#include <cassert>

/*
 Construct OrderedFeature from unsorted feature vector.
 @param _data: unsorted feature vector
 */
OrderedFeature::OrderedFeature(DataMat* data_set, bool _is_categorical,
                               int _feature_index, BoosterConfig* _booster_config):
is_categorical(_is_categorical),
feature_index(_feature_index),
num_data(data_set->num_data),
ordered_data(data_set->num_data, nullptr),
ordered_data_backup(data_set->num_data, nullptr),
booster_config(_booster_config),
leaf_starts(2 * _booster_config->max_leaf + 1, -1),
leaf_ends(2 * _booster_config->max_leaf + 1, -1) {
    
    
        ordered_data.clear();
        ordered_data_backup.clear();
        ordered_data.shrink_to_fit();
        ordered_data_backup.shrink_to_fit();    
        leaf_starts.clear();
        leaf_starts.shrink_to_fit();
        leaf_ends.clear();
        leaf_ends.shrink_to_fit(); 
}

OrderedFeature::OrderedFeature() {}


/*
 Find best split point for a leaf in this feature.
 @param gradients: vector of gradients of current tree. 
 @param hessians: vector of hessians of current tree.
 @param leaf_id: id of the leaf to split.
 @param sum_of_gradients: sum
 */
SplitInfo* OrderedFeature::FindBestSplit(int leaf_id,       
                                         double leaf_gain,
                                         const dvec64 &leaf_sum_up,
                                         const vector<int>& last_split_features) {  
    
    
    return nullptr;
}

void OrderedFeature::Print() const {}

void OrderedFeature::Merge(int left, int right) {   
    std::vector<ValueIndexPair*> buffer;
    
    assert(leaf_starts[left] != leaf_starts[right]);
    
    if(leaf_starts[left] > leaf_starts[right]) {
        int tmp = left;
        left = right;
        right = tmp;
    }
    
    int left_start = leaf_starts[left];
    int right_start = leaf_starts[right];
    int left_end = leaf_ends[left];
    int right_end = leaf_ends[right];
    
    std::vector<int> leaf_counts;
    
    for(int k = 0; k < leaf_starts.size(); ++k) {
        leaf_counts.push_back(leaf_ends[k] - leaf_starts[k]);
    }
    
    int i, j;
    for(i = left_start, j = right_start; i < left_end && j < right_end; ) {
        if(ordered_data[i]->value <= ordered_data[j]->value) {
            buffer.push_back(ordered_data[i]);
            ++i;
        }
        else {
            buffer.push_back(ordered_data[j]);
            ++j;
        }
    }
    
    for(; i < left_end; ++i) {
        buffer.push_back(ordered_data[i]);
    }
    
    for(; j < right_end; ++j) {
        buffer.push_back(ordered_data[j]);
    }
    
    for(int k = right_end - 1; k >= left_end + right_end - right_start; --k) {
        ordered_data[k] = ordered_data[k - (right_end - right_start)];
    }
    
    for(int k = 0; k < leaf_starts.size(); ++k) {
        if(leaf_starts[k] >= left_end && leaf_ends[k] <= right_start) {
            leaf_starts[k] += (right_end - right_start);
            leaf_ends[k] += (right_end - right_start);
        }
    }
    
    assert(int(buffer.size()) == (right_end - right_start + left_end - left_start));    
    
    std::memcpy(ordered_data.data() + left_start, buffer.data(),
                buffer.size() * sizeof(ValueIndexPair*));
    
    
    leaf_ends[left] = left_end + right_end - right_start;
    leaf_starts[left] = left_start;
    leaf_starts[right] = left_start;
    leaf_ends[right] = left_end + right_end - right_start;
    
    for(int k = 0; k < leaf_starts.size(); ++k) {
        assert(leaf_ends[k] - leaf_starts[k] == leaf_counts[k] || k == left || k == right);
    }
}

