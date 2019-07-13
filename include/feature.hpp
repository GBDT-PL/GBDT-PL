//
//  feature.hpp
//  LinearGBM
//


//

#ifndef feature_h
#define feature_h

#include <vector>
#include "split_info.hpp"
#include "booster_config.hpp"
#include <assert.h>
#include <iostream>
#include <algorithm>
#include <cstring>
#include "datamat.hpp"
#include "row_histogram.hpp"
#include "alignment_allocator.hpp"
#include <functional> 

using std::vector;
using std::function;

/*
 Store all the feature values in order, without grouping them by leaves.
 */
class OrderedFeature {
protected:
    
    struct ValueIndexPair {
        
        ValueIndexPair(double _value, double _index): value(_value), index(_index) {}
        
        const double value;
        const int index;
    };
    
    /*
     Starting indices in ordered_data of each leaf.
     */
    std::vector<int> leaf_starts;
    
    /*
     End indices in ordered_data of each leaf.
     */
    std::vector<int> leaf_ends;
    
    bool is_categorical;
    
    int feature_index;
    
    int num_data;
    
    BoosterConfig* booster_config;
    
    int mkl_info;
    int mkl_m;
    int *mkl_ipiv; 
    
    /*
     Sorted feature value with data indices.
     */
    std::vector<ValueIndexPair*> ordered_data;
    
    std::vector<ValueIndexPair*> ordered_data_backup;
    
    std::vector<ValueIndexPair*> left_leaf_data, right_leaf_data;
    
    static bool ComparePair(ValueIndexPair *p1, ValueIndexPair *p2) {   
        return (p1->value < p2->value);
    }
    
public:
    
    bool useful;
    
    double corr;
    double split_gain;
    
    double mkl_time;
    double load_matrix_time;
    
    //virtual function<double()> GetLeafDataIterator(int leaf_id) { return ([] () { return 0.0; }); }
    
    /*
     Construct OrderedFeature from unsorted feature vector.
     @param _data: unsorted feature vector
     */
    OrderedFeature(DataMat* _train_set, bool _is_categorical,
                   int _feature_index, BoosterConfig* _booster_config); 
    
    virtual void BeforeTrain() {}
    
    OrderedFeature();
    
    virtual ~OrderedFeature() {
        ordered_data.clear();
        leaf_starts.clear();
        leaf_ends.clear();
    }
    
    /*
     Deep copy this feature.
     */
    //virtual OrderedFeature* Copy();   
    
    bool IsCategorical() {
        return is_categorical;
    }
    
    virtual int SplitIndex(uint8_t threshold, int leaf_id, int split_start, int split_end,
                           uint8_t *_bit_vector,
                           int *data_indices, int *left_data_indices,
                           int *right_data_indices, int leaf_start) { return 0; }
    
    /*
     Find best split point for a leaf in this feature. Non-Multiclass case
     @param gradients: vector of gradients of current tree.
     @param hessians: vector of hessians of current tree.
     @param leaf_id: id of the leaf to split.
     @param sum_of_gradients: sum
     */
    virtual SplitInfo* FindBestSplit(int leaf_id,
                                     double leaf_gain,
                                     const dvec64 &leaf_sum_up,
                                     const vector<int> &last_split_features = vector<int>());   
    
    
    /*
     Split the feature according to the best split.
     */
    
    void Print() const;
    
    void Merge(int left, int right);
    
    int GetDataNum(int leaf) { return (leaf_ends[leaf] - leaf_starts[leaf]); }  
    
    virtual void PrepareHistogram(int leaf_id, bool use_cache,
                                  RowHistogram* cur, RowHistogram* sibling,
                                  fvec64& bin_gradients,
                                  bool need_augment = false) {};    
    
    
    virtual int GetNumBins() { return 0; }
    
    virtual void Split(int leaf_id, SplitInfo* split, const vector<uint8_t> &bit_vector,
                       int num_left, int  num_leaf_dat, int left_leaf_start, int left_leaf_end,
                       int right_leaf_start, int right_leaf_end) {}
    
    virtual void GetUpdateValues(int leaf_id, SplitInfo* best_split,
                                 double* cur_train_predict_values, const vector<char>& bit_map_vector) {}
    
    virtual void PushIntoBin(double fvalue, int data_idx) {}
    
    virtual void GetLeafSumUp(dvec64&) { }
    
    //virtual void SumUpLeafPredictValues(int leaf_id, const int *data_indices,
    //                                    int leaf_num_data, double k, double *predict_values) {}
    
    virtual void BeforeTrainTree(int iteration, int cur_class, const vector<int> &indices,
                                 const vector<int>& leaf_starts, const vector<int> &leaf_ends) {}
    
    virtual void AfterTrainTree() {}
    
    virtual void Sample(int *sampled_data_indices, int sampled_num_data) {}
    
    virtual uint8_t* get_bins() { return nullptr; }
    
    virtual const uint8_t* GetLocalData(int leaf_id, int leaf_start) { return nullptr; }
    
    virtual void set_hist_info(bool redundantt, int cur_varss, int prev_varss) {}
    
    virtual double* get_feature_values() { return nullptr; }
};

class LinearBinFeature : public OrderedFeature {
protected:
    vector<double> bin_boundaries;
    
    vector<dvec32>* tmp_histograms;
    
    DataMat* train_set;
    const std::vector<int> &global_leaf_starts;
    const std::vector<int> &global_leaf_ends;
    int num_bins;
    dvec32 left_matrix;
    dvec32 left_vec;
    dvec32 right_matrix;
    dvec32 right_vec;
    double *matrix_copy;
    
    vector<uint8_t> *root_data_ptr;     
    
    int cur_num_vars;
    int prev_num_vars;
    
    vector<uint8_t> leaf_local_data;
    vector<uint8_t> copy_tmp; 
    //vector<vector<uint8_t>> leaf_local_data;
    
    double Solve(const dvec32 &matrix,
                 const dvec32 &vec,
                 dvec64 &ks, int n, double l1_reg, int row_size);
    
    void SolveMKL(const dvec32 &matrix,
                  const dvec32 &vec, int n, int row_size,
                  dvec64 &ks);
    
    void LoadMatrix(const dvec32 &histogram,
                    int num_vars, dvec32 &matrix,
                    dvec32 &vec);
    
    bool redundant; 
    
public:
    vector<uint8_t> &data;
    vector<int> &bin_counts;
    vector<double> &bin_values;  
    
    //virtual function<double()> GetLeafDataIterator(int leaf_id);
    
    LinearBinFeature(DataMat* _train_set, bool _is_categorical,
                            int _feature_index, BoosterConfig* _booster_config,
                            const std::vector<int>& _global_leaf_starts, const std::vector<int>& _global_leaf_ends,
                            int _num_vars, vector<uint8_t> & _data,
                            vector<int>& _bin_counts, vector<double>& _bin_values);
    
    virtual int SplitIndex(uint8_t threshold, int leaf_id, int split_start, int split_end,
                   uint8_t *_bit_vector,
                           int *data_indices, int *left_data_indices, int *right_data_indices, int leaf_start);
    
    virtual void BeforeTrain();
    
    virtual void Split(int leaf_id, SplitInfo* split, const vector<uint8_t> &bit_vector,
                       int num_left, int num_leaf_data, int left_leaf_start, int left_leaf_end,
                       int right_leaf_start, int right_leaf_end); 
    
    virtual void PrepareHistogram(int leaf_id, bool use_cache,
                          RowHistogram* cur, RowHistogram* sibling,
                          fvec64 &bin_gradients,
                          bool need_augment = false);
    
    virtual SplitInfo* FindBestSplit(int leaf_id, double leaf_gain,
                             const dvec64 &leaf_sum_up,
                             const vector<int> &last_split_features = vector<int>());   
    
    int GetNumBins() { return num_bins; }   
    
    virtual void PushIntoBin(double fvalue, int data_idx);
    
    const uint8_t* GetLocalData(int leaf_id, int leaf_start);
    
    void GetLeafSumUp(dvec64& leaf_sum_up);
    
    //virtual void SumUpLeafPredictValues(int leaf_id, const int *data_indices,
    //                                    int leaf_num_data, double k, double *predict_values);
    
    virtual void Sample(int *sampled_data_indices, int sampled_num_data);
    
    uint8_t* get_bins() { return data.data(); }
    
    void set_hist_info(bool redundantt, int cur_varss, int prev_varss) {
        redundant = redundantt;
        cur_num_vars = cur_varss;
        prev_num_vars = prev_varss; 
    }
    
    double* get_feature_values() { return bin_values.data(); } 
};

class SparseLinearBinFeature : public LinearBinFeature {
private:
public:
    SparseLinearBinFeature(DataMat* _train_set, bool _is_categorical,
                           int _feature_index, BoosterConfig* _booster_config,
                           const std::vector<int>& _global_leaf_starts, const std::vector<int>& _global_leaf_ends,
                           int _num_vars, vector<uint8_t> & _data,
                           vector<int>& _bin_counts, vector<double>& _bin_values);
    
    void PrepareHistogram(int leaf_id, bool use_cache,
                          RowHistogram* cur, RowHistogram* sibling,
                          fvec64 &bin_gradients,
                          bool need_augment = false);
    
    SplitInfo* FindBestSplit(int leaf_id, double leaf_gain,
                             const dvec64 &leaf_sum_up,
                             const vector<int> &last_split_features = vector<int>());
};

#endif /* feature_h */
