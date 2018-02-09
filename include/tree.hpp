//
//  tree.hpp
//  LinearGBM
//


//

#ifndef tree_h
#define tree_h

#include <vector>
#include <queue>
#include "datamat.hpp"
#include "split_info.hpp"
#include <assert.h>
#include "data_partition.hpp"
#include "feature.hpp"
#include "node.hpp"

using std::vector;

class Tree {
protected:
    DataPartition* data_partition;  
    
    std::vector<SplitInfo*> best_splits;
    
    BoosterConfig* booster_config;
    
    int cur_leaf_num;
    
    int cur_class;
    
    
    //Get the best splits for two newly grown leaves.
    virtual void FindBestSplitForTwoLeaves(int left_leaf_id, int right_leaf_id) = 0;
    
public:
    
    Tree(DataPartition* _data_partition, BoosterConfig* _booster_config, int _cur_class) {
        data_partition = _data_partition;
        booster_config = _booster_config;
        cur_leaf_num = 1;
        cur_class = _cur_class;
    }
    virtual double PredictSingle(const vector<double> &data_point) { return 0.0; }

    virtual void Init() = 0;
    
    virtual void Train() = 0;
    
    virtual void UpdateTrainScore(double *scores) = 0;
    
    virtual void ShrinkToPredictor() = 0;
    
    virtual void Predict(DataMat *test_data, double *scores) = 0;
    
    virtual void PredictTrain(DataPartition *data_partition, DataMat *train_data, double *scores) = 0;
    
    virtual void AfterTrain(DataPartition *data_partition, int cur_class,
                            BoosterConfig *booster_config) {}
    
    virtual ~Tree() {
        for(int i = 0; i < best_splits.size(); ++i) {
            assert(best_splits[i] == nullptr);
        }
    }
};

template <typename NODE_TYPE>
class LeafWiseTree: public Tree {
protected:
    NODE_TYPE *root;
    
    vector<NODE_TYPE*> nodes; 
    
    //Get the best splits for two newly grown leaves
    void FindBestSplitForTwoLeaves(int left_leaf_id, int right_leaf_id);
    
    vector<int> node_split_feature;
    vector<vector<int>> node_linear_model_features;
    vector<double> split_threshold_value;
    vector<uint8_t> split_threshold_bin;
    vector<vector<double>> leaf_ks;
    vector<double> leaf_b;
    vector<int> left_child;
    vector<int> right_child; 
    
    //double PredictSingle(const vector<double> &data_point);
    double PredictTrainSingle(const vector<uint8_t*> &feature_bins, const vector<double*> &feature_values, int index);  
    
public:
    /*
     Constructor.
     @param _data_mat: Training data from the booster.
     @param _booster_config: Configurations of the booster.
     */
    LeafWiseTree(DataPartition* _data_partition, BoosterConfig* _booster_config, int _cur_class);
    
    void UpdateTrainScore(double *scores);
    
    void Init();
    
    /*
     Train a decision tree, using leaf-wise growing.
     @param tree_predictor: output parameter for the tree predictor.
     */
    virtual void Train();
    
    void ShrinkToPredictor();
    
    void Predict(DataMat *test_data, double *scores);
    
    void PredictTrain(DataPartition *data_partition, DataMat *train_data, double *scores);
    
    void AfterTrain(DataPartition *data_partition, int cur_class,
                    BoosterConfig *booster_config);
    double PredictSingle(const vector<double> &data_point);

    virtual ~LeafWiseTree() {
        
    }
};

template <typename NODE_TYPE>
//grow a complete tree level-wise
class LevelWiseTree: public Tree {
protected:
    NODE_TYPE *root;
    
    vector<NODE_TYPE*> nodes; 
    
    //grow one level
    void TrainLevel(int level);
    
    //find the best split and split for the leaf
    void SplitLeaf();
    
    //record which leaves are splittable
    std::vector<bool> is_leaf;
    
    void FindBestSplitForTwoLeaves(int left_leaf_id, int right_leaf_id) {}
public:
    //constructor
    LevelWiseTree(DataPartition* _data_partition, BoosterConfig* _booster_config, int _cur_class);
    
    void UpdateTrainScore(double *scores);
    
    void Init() {} 
    
    //train a level-wise decision tree
    void Train();
    
    void ShrinkToPredictor();
    
    void Predict(DataMat *test_data, double *scores) {}
    
    void PredictTrain(DataPartition *data_partition, DataMat *train_data, double *scores) {}
    
    void AfterTrain(DataPartition *data_partition, int cur_class,
                    BoosterConfig *booster_config) {}
    
    //destructor
    virtual ~LevelWiseTree() {}
};

#endif /* tree_h */
