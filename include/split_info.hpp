//
//  split_info.hpp
//  LinearGBM
//


//

#ifndef split_info_hpp
#define split_info_hpp

#include <vector>
#include <iostream>

using std::vector;

class SplitInfo {
protected:
    SplitInfo(int _feature_index, uint8_t _threshold, double _threshold_value, double _gain,
              
              double _left_gain, double _right_gain):
    feature_index(_feature_index), threshold(_threshold),
    threshold_value(_threshold_value), gain(_gain),
    left_gain(_left_gain), right_gain(_right_gain) {}
public:
    int feature_index;
    double threshold;
    double gain;
    int left_leaf_id;
    int right_leaf_id;
    double threshold_value;
    
    double left_gain, right_gain;
    
    void SetChildrenID(int _left_leaf_id, int _right_leaf_id) {
        left_leaf_id = _left_leaf_id;
        right_leaf_id = _right_leaf_id;
    }
    
    virtual ~SplitInfo() {}
};

class ConstantSplitInfo: public SplitInfo {
    
public:
    ConstantSplitInfo(int _feature_index,
                      uint8_t _threshold, double _threshold_value,
                      double _gain,
                      double _left_leaf_predict_value,
                      double _right_leaf_predict_value,
                      double _left_gain,
                      double _right_gain): SplitInfo(_feature_index, _threshold, _threshold_value,
                                                     _gain,
                                                     _left_gain, _right_gain),
    left_leaf_predict_value(_left_leaf_predict_value),
    right_leaf_predict_value(_right_leaf_predict_value) {}
    
    ~ConstantSplitInfo() {}
    
    double left_leaf_predict_value;
    double right_leaf_predict_value;
};

class MultipleLinearSplitInfo: public SplitInfo {
public:
    MultipleLinearSplitInfo(int _feature_index, uint8_t _threshold,
                            double _threshold_value, double _gain,
                            double _left_gain, double _right_gain,
                            vector<double> _left_ks, double _left_b,
                            vector<double> _right_ks, double _right_b,
                            vector<int> _last_feature_indices):
    SplitInfo(_feature_index, _threshold, _threshold_value, _gain,
              _left_gain, _right_gain) {
        left_ks = _left_ks;
        right_ks = _right_ks;
        left_b = _left_b;
        right_b = _right_b;
        last_feature_indices = _last_feature_indices;
    }
    
    vector<double> left_ks, right_ks;
    double left_b, right_b;
    vector<int> last_feature_indices;
};

class AdditiveLinearSplitInfo: public SplitInfo {
public:
    AdditiveLinearSplitInfo(int _feature_index, uint8_t _threshold,
                            double _threshold_value, double _gain,
                            double _left_gain, double _right_gain,
                            double _left_k, double _left_b, double _right_k, double _right_b,
                            double _left_sum_of_bin_bin_hessians, double _right_sum_of_bin_bin_hessians,
                            double _left_sum_of_hessian_pred_bin, double _right_sum_of_hessian_pred_bin,
                            double _left_sum_of_gradient_bin, double _right_sum_of_gradient_bin,
                            double _left_sum_of_hessian_pred, double _right_sum_of_hessian_pred,
                            double _left_sum_of_bin_hessians, double _right_sum_of_bin_hessians,
                            bool _from_sparse):
    SplitInfo(_feature_index, _threshold, _threshold_value, _gain,
              _left_gain, _right_gain) {
        left_k = _left_k;
        left_b = _left_b;
        right_k = _right_k;
        right_b = _right_b;
        left_sum_of_bin_bin_hessians = _left_sum_of_bin_bin_hessians;
        right_sum_of_bin_bin_hessians = _right_sum_of_bin_bin_hessians;
        left_sum_of_hessian_pred_bin = _left_sum_of_hessian_pred_bin;
        right_sum_of_hessian_pred_bin = _right_sum_of_hessian_pred_bin;
        left_sum_of_gradient_bin = _left_sum_of_gradient_bin;
        right_sum_of_gradient_bin = _right_sum_of_gradient_bin;
        left_sum_of_hessian_pred = _left_sum_of_hessian_pred;
        right_sum_of_hessian_pred = _right_sum_of_hessian_pred;
        left_sum_of_bin_hessians = _left_sum_of_bin_hessians;
        right_sum_of_bin_hessians = _right_sum_of_bin_hessians;
        from_sparse = _from_sparse;
    }
    
    double left_k, right_k;
    double left_b, right_b;
    double left_sum_of_bin_bin_hessians;
    double right_sum_of_bin_bin_hessians;
    double left_sum_of_hessian_pred_bin;
    double right_sum_of_hessian_pred_bin;
    double left_sum_of_gradient_bin;
    double right_sum_of_gradient_bin;
    double left_sum_of_hessian_pred;
    double right_sum_of_hessian_pred;
    double left_sum_of_bin_hessians;
    double right_sum_of_bin_hessians;
    
    bool from_sparse;
};

#endif /* split_info_hpp */
