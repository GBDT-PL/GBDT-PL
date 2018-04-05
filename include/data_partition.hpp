//
//  data_partition.hpp
//  LinearGBM
//


//

#ifndef data_partition_hpp
#define data_partition_hpp

#include "data_partition.hpp"
#include "datamat.hpp"
#include <vector>
#include "booster_config.hpp"
#include "feature.hpp"
#include <queue>
#include "row_histogram.hpp"
#include "split_info.hpp"
#include <set>
#include <queue>
#include "alignment_allocator.hpp"
#include "objective.hpp"

using std::vector;
using std::queue;
using std::set;
using std::cout;
using std::endl;

class DataPartition {
private:
    DataMat* train_set;
    
    vector<int> leaf_starts, leaf_ends; 
    
    BoosterConfig* booster_config;
    
    vector<OrderedFeature*> features;
    
    int least_bin_feature;
    
    vector<int> data_indices;
    
    vector<int> data_indices_in_left;
    
    vector<int> data_indices_in_right;
    
    fvec64 bin_gradients;
    
    fvec64 &all_gradients;
    
    fvec64 pred_values;
    
    int sampled_num_data;
    
    vector<int> left_copy_offsets;
    
    vector<int> right_copy_offsets;
    
    vector<uint8_t> bit_vector;
    
    vector<vector<RowHistogram*>*> histogram_repo;
    
    int first_unused_histogram; 
    
    vector<int> useful_features;
    
    int num_data; 
    int num_data_aligned; 
    
    dvec64 leaf_sum_up;
    
    RowHistogram* GetHistogram(int leaf_id, int feature_id);
    
    void PrepareHistogramSize(int leaf_id, bool use_cache);
    
    void PushDataIntoFeatures();
    
public:
    DataPartition(DataMat* _train_set, BoosterConfig* _booster_config,  
                  fvec64 &gradients);
    
    void Split(int leaf_id, SplitInfo* split, int cur_class);
    
    void BeforeTrainTree(int cur_class, int iteration);
    
    void BeforeTrain();
    
    vector<RowHistogram*>* GetHistogramFromPool();                              
    
    vector<int>& get_data_indices() { return data_indices; }
    
    const vector<int>& get_leaf_starts() { return leaf_starts; }
    
    const vector<int>& get_leaf_ends() { return leaf_ends; }
    
    const vector<OrderedFeature*>& get_features() { return features; }
    
    fvec64& get_bin_gradients() { return bin_gradients; }
    
    float *get_all_gradients(int cur_class) { return all_gradients.data() + 2 * cur_class * num_data_aligned; }
    
    const vector<int> &get_useful_features() { return useful_features; }
    
    int get_num_data() { return num_data; }
    
    dvec64 &get_leaf_sum_up() { return leaf_sum_up; }
    
    uint8_t* get_feature_bin(int feature_index) { return features[feature_index]->get_bins(); }
    
    double* get_feature_values(int feature_index) { return features[feature_index]->get_feature_values(); }
    
    const vector<uint8_t>& get_bit_vector() { return bit_vector; }
    
    void reset_pred_values() { pred_values.clear(); pred_values.resize(train_set->num_data, 0.0); }
    
    fvec64& get_pred_values() { return pred_values; }
    
    const vector<int>& get_left_copy_offset() { return left_copy_offsets; }
    const vector<int>& get_right_copy_offset() { return right_copy_offsets; }    
    
    void LeafSumUp(); 
    
    int GetLeafSize(int leaf_id) {
        return leaf_ends[leaf_id] - leaf_starts[leaf_id];
    }
    
    void PrintAllTime();
    
    void AfterTrainTree();
    
    void Sample(float *gradients_ptrr, int num_dataa);
};


#endif /* data_partition_hpp */
