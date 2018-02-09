//
//  tree_predictor.cpp
//  LinearGBM
//


//

#include <stdio.h>
#include "tree.hpp"

template <>
void LeafWiseTree<LinearNode>::ShrinkToPredictor() {
    //already shrink
    if(nodes.size() == 0) {
        return;
    }
    
    int num_nodes = 2 * booster_config->max_leaf + 1;
    node_split_feature.resize(num_nodes, -1);
    node_linear_model_features.resize(num_nodes);
    leaf_ks.resize(num_nodes);
    leaf_b.resize(num_nodes, 0.0);
    split_threshold_value.resize(num_nodes, 0.0);
    split_threshold_bin.resize(num_nodes, 0);               
    left_child.resize(num_nodes, -1);
    right_child.resize(num_nodes, -1);
    root->FillInTreePredictor(node_split_feature, node_linear_model_features,
                              leaf_ks, leaf_b, split_threshold_value, split_threshold_bin, left_child, right_child);
    delete root;
    nodes.clear();
    nodes.shrink_to_fit();
}

template <>
void LeafWiseTree<ConstantNode>::ShrinkToPredictor() {
    
}

template <>
double LeafWiseTree<LinearNode>::PredictSingle(const vector<double> &data_point) {
    int current_leaf = 0;
    while(node_split_feature[current_leaf] != -1) {
        double split_threshold = split_threshold_value[current_leaf];
        
        int split_feature_index = node_split_feature[current_leaf];
        if(data_point[split_feature_index] <= split_threshold) {
            current_leaf = left_child[current_leaf];
        }
        else {
            current_leaf = right_child[current_leaf];
        }
    }
    
    double result = 0.0;
    for(int i = 0; i < node_linear_model_features[current_leaf].size(); ++i) {
        result += leaf_ks[current_leaf][i] * data_point[node_linear_model_features[current_leaf][i]];
    }
    result += leaf_b[current_leaf];
    
    return booster_config->learning_rate * result;
}

template <>
double LeafWiseTree<ConstantNode>::PredictSingle(const vector<double> &data_point) { return 0.0; }

template <>     
void LeafWiseTree<LinearNode>::Predict(DataMat *test_data, double *scores) {
    const vector<vector<double>> &data = test_data->unbined_data;
#pragma omp parallel for schedule(static) num_threads(24)
    for(int i = 0; i < data.size(); ++i) {
        scores[i] += PredictSingle(data[i]);
    }
}

template <>
void LeafWiseTree<ConstantNode>::Predict(DataMat *test_data, double *scores) {
    
}

template <>
double LeafWiseTree<LinearNode>::PredictTrainSingle(const vector<uint8_t*> &feature_bins,
                                                    const vector<double*> &feature_values,
                                                    int index) {
    int current_leaf = 0;
    while(node_split_feature[current_leaf] != -1) {
        uint8_t split_threshold = split_threshold_bin[current_leaf];
        
        int split_feature_index = node_split_feature[current_leaf];
        if(feature_bins[split_feature_index][index] <= split_threshold) {
            current_leaf = left_child[current_leaf];
        }
        else {
            current_leaf = right_child[current_leaf];
        }
    }
    
    double result = 0.0;
    for(int i = 0; i < node_linear_model_features[current_leaf].size(); ++i) {
        int feature_id = node_linear_model_features[current_leaf][i];
        result += leaf_ks[current_leaf][i] * feature_values[feature_id][feature_bins[feature_id][index]];
    }
    result += leaf_b[current_leaf];
    
    return booster_config->learning_rate * result;      
}

template <>
double LeafWiseTree<ConstantNode>::PredictTrainSingle(const vector<uint8_t*> &feature_bins,
                                                      const vector<double*> &feature_values,
                                                      int index) {
    return 0.0;
}

template <>
void LeafWiseTree<LinearNode>::PredictTrain(DataPartition *data_partition, DataMat *train_data, double *scores) {
    vector<uint8_t*> feature_bins(train_data->num_feature, nullptr);
    vector<double*> feature_values(train_data->num_feature, nullptr);
    
#pragma omp parallel for schedule(static) num_threads(booster_config->num_threads)
    for(int i = 0; i < train_data->num_feature; ++i) {
        feature_bins[i] = data_partition->get_feature_bin(i);
        feature_values[i] = data_partition->get_feature_values(i);
    }
    
#pragma omp parallel for schedule(static) num_threads(booster_config->num_threads)
    for(int i = 0; i < train_data->num_data; ++i) {
        scores[i] += PredictTrainSingle(feature_bins, feature_values, i);
    }
}

template <>
void LeafWiseTree<ConstantNode>::PredictTrain(DataPartition *data_partition, DataMat *train_data, double *scores) {}

template <>
void LeafWiseTree<AdditiveLinearNode>::ShrinkToPredictor() {
    if(nodes.size() == 0) {
        return;
    }
    
    int num_nodes = 2 * booster_config->max_leaf + 1;
    node_split_feature.resize(num_nodes, -1);
    node_linear_model_features.resize(num_nodes);
    leaf_ks.resize(num_nodes);
    leaf_b.resize(num_nodes, 0.0);
    split_threshold_value.resize(num_nodes, 0.0);
    split_threshold_bin.resize(num_nodes, 0);
    left_child.resize(num_nodes, -1);
    right_child.resize(num_nodes, -1);
    root->FillInTreePredictor(node_split_feature, node_linear_model_features,
                              leaf_ks, leaf_b, split_threshold_value, split_threshold_bin, left_child, right_child);
    delete root;
    nodes.clear();
    nodes.shrink_to_fit();
}

template <>
double LeafWiseTree<AdditiveLinearNode>::PredictSingle(const vector<double> &data_point) {
    int current_leaf = 0;
    while(node_split_feature[current_leaf] != -1) {
        double split_threshold = split_threshold_value[current_leaf];
        
        int split_feature_index = node_split_feature[current_leaf];
        if(data_point[split_feature_index] <= split_threshold) {
            current_leaf = left_child[current_leaf];
        }
        else {
            current_leaf = right_child[current_leaf];
        }
    }
    
    double result = 0.0;
    for(int i = 0; i < node_linear_model_features[current_leaf].size(); ++i) {
        result += leaf_ks[current_leaf][i] * data_point[node_linear_model_features[current_leaf][i]];
    }
    result += leaf_b[current_leaf];
    
    return booster_config->learning_rate * result;
}

template <>
double LeafWiseTree<AdditiveLinearNode>::PredictTrainSingle(const vector<uint8_t*> &feature_bins,
                                                            const vector<double*> &feature_values,
                                                            int index) {
    int current_leaf = 0;
    while(node_split_feature[current_leaf] != -1) {
        uint8_t split_threshold = split_threshold_bin[current_leaf];
        
        int split_feature_index = node_split_feature[current_leaf];
        if(feature_bins[split_feature_index][index] <= split_threshold) {
            current_leaf = left_child[current_leaf];
        }
        else {
            current_leaf = right_child[current_leaf];
        }
    }
    
    double result = 0.0;
    for(int i = 0; i < node_linear_model_features[current_leaf].size(); ++i) {
        int feature_id = node_linear_model_features[current_leaf][i];
        result += leaf_ks[current_leaf][i] * feature_values[feature_id][feature_bins[feature_id][index]];
    }
    result += leaf_b[current_leaf];
    
    return booster_config->learning_rate * result;
}

template <>
void LeafWiseTree<AdditiveLinearNode>::Predict(DataMat *test_data, double *scores) {
    const vector<vector<double>> &data = test_data->unbined_data;
#pragma omp parallel for schedule(static) num_threads(booster_config->num_threads)
    for(int i = 0; i < data.size(); ++i) {
        scores[i] += PredictSingle(data[i]);
    }
}

template <>
void LeafWiseTree<AdditiveLinearNode>::PredictTrain(DataPartition *data_partition,
                                                    DataMat *train_data, double *scores) {
    vector<uint8_t*> feature_bins(train_data->num_feature, nullptr);
    vector<double*> feature_values(train_data->num_feature, nullptr);                           
    
#pragma omp parallel for schedule(static) num_threads(booster_config->num_threads)
    for(int i = 0; i < train_data->num_feature; ++i) {
        feature_bins[i] = data_partition->get_feature_bin(i);
        feature_values[i] = data_partition->get_feature_values(i);
    }
    
#pragma omp parallel for schedule(static) num_threads(booster_config->num_threads)
    for(int i = 0; i < train_data->num_data; ++i) {
        scores[i] += PredictTrainSingle(feature_bins, feature_values, i);
    }
}
