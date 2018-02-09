//
//  leaf.hpp
//  LinearGBMVector
//


//

#ifndef node_h
#define node_h

#include <vector>
#include "split_info.hpp"
#include "datamat.hpp"
#include "feature.hpp"
#include "booster_config.hpp"
#include "data_partition.hpp"

using std::vector;

class Node {
protected:
    int leaf_id;
    int depth;
    
public:
    virtual void GetAllLeaves(vector<Node*> &leaves) = 0;
    
    virtual void UpdateTrainScore(const vector<int> &indices, double *score,    
                                  const vector<OrderedFeature*> &features,      
                                  const BoosterConfig *booster_config,
                                  const vector<int> &leaf_starts,
                                  const vector<int> &leaf_ends) = 0;
    
    virtual SplitInfo* FindBestSplit(DataPartition *data_partition, BoosterConfig *booster_config, int cur_class) = 0;
    
    virtual void AfterTrain(DataPartition *data_partition, int cur_class,
                            BoosterConfig *booster_config) = 0;
};

class LinearNode : public Node {
protected:
    vector<double> leaf_k;
    double leaf_b;
    vector<int> linear_model_features;
    LinearNode(const vector<double> &leaf_k, const double &leaf_b,
               vector<int> linear_model_features, int leaf_id, bool smaller_node,
               vector<RowHistogram*>* histogram, LinearNode *parent, double gain); 
    
    void set_sibling(LinearNode* siblingg) { sibling = siblingg; }
    
    bool smaller_node;
    
    double gain;
    int split_index;
    uint8_t split_bin;
    double split_threshold;
    
    LinearNode *left_child;
    LinearNode *right_child;
    
    vector<RowHistogram*>* histogram;
    
    LinearNode *sibling;
    LinearNode *parent; 
    
    virtual void PrepareBinGradients(DataPartition *data_partition, BoosterConfig *booster_config, int cur_class);
    
    virtual void PrepareHistograms(DataPartition *data_partition, BoosterConfig *booster_config, int cur_class);
    
    virtual SplitInfo* FindBestSplits(DataPartition *data_partition, BoosterConfig *booster_config, int cur_class);
    
public:
    LinearNode(DataPartition *data_partition);
    virtual void Split(SplitInfo *split, LinearNode **out_left, LinearNode **out_right,
               int max_var, DataPartition *data_partition, int num_threads);
    virtual void UpdateTrainScore(const vector<int> &indices, double *score,
                          const vector<OrderedFeature*> &features,
                          const BoosterConfig *booster_config,
                          const vector<int> &leaf_starts,
                          const vector<int> &leaf_ends);
    virtual SplitInfo* FindBestSplit(DataPartition *data_partition, BoosterConfig *booster_config, int cur_class);
    
    virtual void FillInTreePredictor(vector<int> &split_index, vector<vector<int>> &linear_features,
                             vector<vector<double>> &leaf_ks, vector<double> &leaf_b,
                             vector<double> &split_threshold_value, vector<uint8_t> &split_threshold_bin,
                             vector<int> &left_child, vector<int> &right_child);
    
    virtual void GetAllLeaves(vector<Node*> &levaes);
    
    int get_feature_num() { return  static_cast<int>(linear_model_features.size()); }   
    
    vector<RowHistogram*>* get_histogram() { return histogram; }
    
    virtual void AfterTrain(DataPartition *data_partition, int cur_class,
                            BoosterConfig *booster_config) {}
    
    virtual ~LinearNode(); 
};

class AdditiveLinearNode : public LinearNode {
protected:
    
    AdditiveLinearNode *left_child;
    AdditiveLinearNode *right_child;
    
    AdditiveLinearNode *sibling;    
    AdditiveLinearNode *parent;
    
    //fvec64 pred_values;
    
    virtual void PrepareBinGradients(DataPartition *data_partition, BoosterConfig *booster_config, int cur_class);
    
    virtual void PrepareHistograms(DataPartition *data_partition, BoosterConfig *booster_config, int cur_class);
    
    virtual SplitInfo* FindBestSplits(DataPartition *data_partition, BoosterConfig *booster_config, int cur_class);
    
    AdditiveLinearNode(const vector<double> &leaf_k, const double &leaf_b,
               vector<int> linear_model_features, int leaf_id, bool smaller_node,
               vector<RowHistogram*>* histogram, AdditiveLinearNode *parent, double gain);
    
    void set_sibling(AdditiveLinearNode* siblingg) { sibling = siblingg; }
    
    //fvec64& get_pred_values() { return pred_values; }     
    
public:
    AdditiveLinearNode(DataPartition *data_partition);
    
    void Split(SplitInfo *split, AdditiveLinearNode **out_left, AdditiveLinearNode **out_right,
               int max_var, DataPartition *data_partition, int num_threads);
    
    void FillInTreePredictor(vector<int> &node_split_feature_index,
                             vector<vector<int> > &node_linear_features,
                             vector<vector<double> > &leaf_ks, vector<double> &leaf_bs,
                             vector<double> &split_threshold_value, vector<uint8_t> &split_threshold_bin,
                             vector<int> &left_childs,
                             vector<int> &right_childs);
    
    void AfterTrain(DataPartition *data_partition, int cur_class,
                    BoosterConfig *booster_config);
    
    void UpdateTrainScore(const vector<int> &indices, double *score,
                          const vector<OrderedFeature*> &features,
                          const BoosterConfig *booster_config,
                          const vector<int> &leaf_starts,
                          const vector<int> &leaf_ends);
    
    void GetAllLeaves(vector<Node *> &leaves);
};

class ConstantNode : public Node {
protected:
    vector<double> leaf_bs;
    
    ConstantNode *left_child;
    ConstantNode *right_child;
    virtual void PrepareBinGradients() {}
public:
    ConstantNode(DataPartition *data_partition) {}      
    void Split(SplitInfo *split, ConstantNode **out_left, ConstantNode **out_right, 
               int max_var, DataPartition *data_partition, int num_threads) {}
    void UpdateTrainScore(const vector<int> &indices, double *score,
                          const vector<OrderedFeature*> &features,
                          const BoosterConfig *booster_config,
                          const vector<int> &leaf_starts,
                          const vector<int> &leaf_ends) {}
    virtual SplitInfo* FindBestSplit(DataPartition *data_partition, BoosterConfig *booster_config, int cur_class) { return nullptr; }
    
    void GetAllLeaves(vector<Node*> &levaes) {}
    
    void AfterTrain(DataPartition *data_partition, int cur_class,
                    BoosterConfig *booster_config) {}
};

#endif /* node_h */
