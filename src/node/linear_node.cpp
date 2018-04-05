//
//  linear_leaf.cpp
//  LinearGBMVector
//


//

#include <stdio.h>
#include "node.hpp"
#include <omp.h>

//create root node
LinearNode::LinearNode(DataPartition *data_partition) {
    left_child = nullptr;
    right_child = nullptr;
    parent = nullptr;
    sibling = nullptr;
    leaf_id = 0;
    depth = 0;
    smaller_node = false;
    histogram = data_partition->GetHistogramFromPool();
    gain = 0.0;
    split_index = -1; 
}   

LinearNode::LinearNode(const vector<double> &ks, const double &b,
                       vector<int> linear_model_featuress, int leaf_idd, bool smaller_nodee,
                       vector<RowHistogram*>* histogramm,
                       LinearNode* parentt, double gainn) {
    leaf_id = leaf_idd;
    leaf_k = ks;
    leaf_b = b;
    linear_model_features = linear_model_featuress;
    left_child = nullptr;
    right_child = nullptr;
    smaller_node = smaller_nodee;
    histogram = histogramm;
    parent = parentt;
    gain = gainn;
    split_index = -1; 
}

void LinearNode::UpdateTrainScore(const vector<int> &indices, double *scores,
                                  const vector<OrderedFeature*> &features,
                                  const BoosterConfig *booster_config,  
                                  const vector<int> &leaf_starts,
                                  const vector<int> &leaf_ends) {
    if(left_child != nullptr) {
        left_child->UpdateTrainScore(indices, scores, features, booster_config, leaf_starts, leaf_ends);
        right_child->UpdateTrainScore(indices, scores, features, booster_config, leaf_starts, leaf_ends);   
    }
    else {
        int leaf_num_features = static_cast<int>(linear_model_features.size());
        vector<const unsigned char*> data_bins(leaf_num_features);
        vector<double*> bin_valuess(leaf_num_features);
        for(int j = 0; j < leaf_num_features; ++j) {
            data_bins[j] = (static_cast<LinearBinFeature*>(features[linear_model_features[j]]))->GetLocalData(leaf_id, leaf_starts[leaf_id]);   
            bin_valuess[j] = features[linear_model_features[j]]->get_feature_values();
        }
        
        int leaf_start = leaf_starts[leaf_id], leaf_end = leaf_ends[leaf_id];
        const int *local_data_indices = indices.data() + leaf_start;
        
        double lr = booster_config->learning_rate;
#pragma omp parallel for schedule(static) num_threads(booster_config->num_threads)
        for(int j = 0; j < leaf_end - leaf_start; ++j) {
            int index = local_data_indices[j];
            double result = 0.0;
            for(int k = 0; k < leaf_num_features; ++k) {
                result += bin_valuess[k][data_bins[k][j]] * leaf_k[k];
            }
            result += leaf_b;
            scores[index] += lr * result;
        }
    }
}

void LinearNode::GetAllLeaves(vector<Node *> &leaves) {     
    if(left_child == nullptr && right_child == nullptr) {
        leaves.push_back(this);
    }
    else {
        left_child->GetAllLeaves(leaves);
        right_child->GetAllLeaves(leaves);
    }
}

void LinearNode::Split(SplitInfo *split, LinearNode **out_left,
                       LinearNode **out_right, int max_var, DataPartition *data_partition, int num_threads) {
    MultipleLinearSplitInfo* best_split = static_cast<MultipleLinearSplitInfo*>(split);         
    split_index = best_split->feature_index;
    split_threshold = best_split->threshold_value;
    split_bin = static_cast<uint8_t>(best_split->threshold);    
    bool redundant = false;
    vector<int> to_left = linear_model_features, to_right = linear_model_features;              
    for(int i = 0; i < linear_model_features.size(); ++i) {
        if(linear_model_features[i] == split->feature_index) {
            redundant = true;
            break;
        }
    }
    if(!redundant && linear_model_features.size() < max_var) {  
        to_left.push_back(split->feature_index);
        to_right.push_back(split->feature_index);
    }
    
    const vector<int> &leaf_starts = data_partition->get_leaf_starts();
    const vector<int> &leaf_ends = data_partition->get_leaf_ends();
    
    int left_leaf_start = leaf_starts[best_split->left_leaf_id];
    int right_leaf_start = leaf_starts[best_split->right_leaf_id];
    int left_leaf_end = leaf_ends[best_split->left_leaf_id];
    int right_leaf_end = leaf_ends[best_split->right_leaf_id];
    
    bool left_smaller = (left_leaf_end - left_leaf_start) <= (right_leaf_end - right_leaf_start);
    
    if(left_smaller) {
        left_child = new LinearNode(best_split->left_ks, best_split->left_b, to_left,
                                best_split->left_leaf_id, left_smaller,
                                    nullptr, this, best_split->left_gain);
        right_child = new LinearNode(best_split->right_ks, best_split->right_b, to_right,
                                 best_split->right_leaf_id, !left_smaller,
                                     histogram, this, best_split->right_gain);
    }
    else {
        left_child = new LinearNode(best_split->left_ks, best_split->left_b, to_left,   
                                    best_split->left_leaf_id, left_smaller,
                                    histogram, this, best_split->left_gain);
        right_child = new LinearNode(best_split->right_ks, best_split->right_b, to_right,
                                     best_split->right_leaf_id, !left_smaller,              
                                     nullptr, this, best_split->right_gain);
    }
    left_child->set_sibling(right_child);   
    right_child->set_sibling(left_child);
    *out_left = left_child;
    *out_right = right_child;   
}

void LinearNode::PrepareBinGradients(DataPartition *data_partition, BoosterConfig *booster_config, int cur_class) {
    int leaf_start = data_partition->get_leaf_starts()[leaf_id];
    int leaf_end = data_partition->get_leaf_ends()[leaf_id];
    float* local_gradients = data_partition->get_all_gradients(cur_class);
    int leaf_num_data = leaf_end - leaf_start;
    int num_vars = static_cast<int>(linear_model_features.size());
    
    const int *local_data_indices = data_partition->get_data_indices().data() + leaf_start;
    const vector<OrderedFeature*> features = data_partition->get_features();
    
    fvec64 &bin_gradients = data_partition->get_bin_gradients();
    
    if(leaf_id == 0) {
        if(bin_gradients.size() < 2 * leaf_num_data) {
            bin_gradients.resize(2 * leaf_num_data, 0.0);           
        }
#pragma omp parallel for schedule(static) num_threads(booster_config->num_threads)
        for(int i = 0; i < leaf_end - leaf_start; ++i) {
            int index = local_data_indices[i];
            bin_gradients[2 * i] = local_gradients[2 * index];
            bin_gradients[2 * i + 1] = local_gradients[2 * index + 1];                              
        }
        
        for(int i = 2 * (leaf_end - leaf_start); i < 2 * (leaf_end - leaf_start) + 32 &&
                i < bin_gradients.size(); ++i) {
            bin_gradients[i] = 0.0; 
        }
        
        return;
    }
    
    if(smaller_node) {
        int row_size = (2 + num_vars * 3 + num_vars * (num_vars - 1) / 2);
        
        if(bin_gradients.size() < row_size * leaf_num_data) {
            bin_gradients.resize(row_size * leaf_num_data);
        }
        
        vector<double*> feature_bin_values(num_vars, nullptr);
        vector<const uint8_t*> feature_bins(num_vars, nullptr);
        
        for(int i = 0; i < num_vars; ++i) {
            feature_bins[i] = features[linear_model_features[i]]->GetLocalData(leaf_id, leaf_start);
            feature_bin_values[i] = features[linear_model_features[i]]->get_feature_values();
        }
        
        float *bin_gradients_ptr = bin_gradients.data();
#pragma omp parallel for schedule(static) num_threads(booster_config->num_threads)
        for(int i = 0; i < leaf_end - leaf_start; ++i) {
            float *local_bin_gradient = bin_gradients_ptr + i * row_size;
            int index = local_data_indices[i] << 1;
            const float gradient = local_gradients[index];
            const float hessian = local_gradients[index + 1];
            local_bin_gradient[0] = gradient;
            local_bin_gradient[1] = hessian;
            local_bin_gradient += 2;
            
            for(int j = 0; j < num_vars; ++j) {
                const float bin_value = feature_bin_values[j][feature_bins[j][i]];
                local_bin_gradient[0] = bin_value * gradient;
                double bin_hessian = bin_value * hessian;
                local_bin_gradient[1] = bin_hessian;
                local_bin_gradient[2] = bin_value * bin_hessian;
                
                for(int k = 0; k < j; ++k) {
                    const float bin_value_2 = feature_bin_values[k][feature_bins[k][i]];    
                    local_bin_gradient[3 + k] = bin_hessian * bin_value_2;
                }
                local_bin_gradient += 3 + j;
            }
        }
    }
    else if(parent->linear_model_features.size() < linear_model_features.size()) {
        int row_size = num_vars + 2;
        
        if(bin_gradients.size() < row_size * leaf_num_data) {
            bin_gradients.resize(row_size * leaf_num_data);
        }
        
        vector<double*> feature_bin_values(num_vars, nullptr);
        vector<const uint8_t*> feature_bins(num_vars, nullptr);
        
        for(int i = 0; i < num_vars; ++i) {
            feature_bins[i] = features[linear_model_features[i]]->GetLocalData(leaf_id, leaf_start);
            feature_bin_values[i] = features[linear_model_features[i]]->get_feature_values();
        }
        
        float *bin_gradient_ptr = bin_gradients.data();
#pragma omp parallel for schedule(static) num_threads(booster_config->num_threads)
        for(int i = 0; i < leaf_end - leaf_start; ++i) {
            float* local_bin_gradient = bin_gradient_ptr + i * row_size;
            
            const float bin_value = feature_bin_values.back()[feature_bins.back()[i]];
            int index = local_data_indices[i] << 1;
            const float gradient = local_gradients[index];
            const float hessian = local_gradients[index + 1];
            local_bin_gradient[0] = bin_value * gradient;
            double bin_hessian = bin_value * hessian;
            local_bin_gradient[1] = bin_hessian;
            local_bin_gradient[2] = bin_value * bin_hessian;
            for(int j = 0; j < num_vars - 1; ++j) {
                const float bin_value_2 = feature_bin_values[j][feature_bins[j][i]]; 
                local_bin_gradient[j + 3] = bin_hessian * bin_value_2;
            }
        }
    }
}

void LinearNode::PrepareHistograms(DataPartition *data_partition, BoosterConfig *booster_config, int cur_class) {
    const vector<int>& useful_features = data_partition->get_useful_features();
    vector<OrderedFeature*> features = data_partition->get_features();
    fvec64 &bin_gradients = data_partition->get_bin_gradients();
    bool need_augment = true;
    if(parent != nullptr && parent->linear_model_features.size() == linear_model_features.size()) {
        need_augment = false;
    }
    
    if(histogram == nullptr) {
        histogram = data_partition->GetHistogramFromPool();     
    }
    
#pragma omp parallel for schedule(guided) num_threads(booster_config->num_threads)
    for(int i = 0; i < useful_features.size(); ++i) {
        int fid = useful_features[i];   
        
        RowHistogram* cur_hist = (*histogram)[fid];
        RowHistogram* sibling_hist = nullptr;
        if(leaf_id != 0) {
            sibling_hist = (*(sibling->histogram))[fid];                                                    
        }
        
        if(leaf_id == 0) {
            features[fid]->set_hist_info(false, 1, 0);
        }
        else {
            bool redundant = false;
            for(int i = 0; i < linear_model_features.size(); ++i) {
                if(fid == linear_model_features[i]) {
                    redundant = true;
                    break;
                }
            }
            
            redundant |= (linear_model_features.size() == booster_config->num_vars);
            
            int cur_num_vars = 0, prev_num_vars = 0;
            if(!redundant) {
                cur_num_vars = static_cast<int>(linear_model_features.size()) + 1;
                prev_num_vars = cur_num_vars - 1;
            }
            else {
                cur_num_vars = static_cast<int>(linear_model_features.size());
                prev_num_vars = cur_num_vars;
            }
            features[fid]->set_hist_info(redundant, cur_num_vars, prev_num_vars); 
        }
        
        features[fid]->PrepareHistogram(leaf_id, !smaller_node,
                                        cur_hist,
                                        sibling_hist,
                                        bin_gradients,
                                        need_augment);  
    }
    data_partition->LeafSumUp();
}

SplitInfo* LinearNode::FindBestSplits(DataPartition *data_partition, BoosterConfig *booster_config, int cur_class) {
    const vector<int>& useful_features = data_partition->get_useful_features();
    const vector<OrderedFeature*> features = data_partition->get_features();
    dvec64 &leaf_sum_up = data_partition->get_leaf_sum_up();
    int num_threads = booster_config->num_threads;
    vector<SplitInfo*> thread_best_splits(num_threads, nullptr);    
#pragma omp parallel for schedule(guided) num_threads(num_threads)
    for(int i = 0; i < useful_features.size(); ++i) {
        int fid = useful_features[i];
        SplitInfo* split = nullptr;
        
        split = features[fid]->FindBestSplit(leaf_id, gain,
                                             leaf_sum_up,
                                             linear_model_features);    
        
        int tid = omp_get_thread_num();
        
        if(thread_best_splits[tid] == nullptr ||
           (split != nullptr && thread_best_splits[tid]->gain < split->gain)) {
            thread_best_splits[tid] = split;
        }
        else {
            if(split != nullptr) {
                delete split;
            }
        }
    }
    
    SplitInfo* best_split = nullptr;
    
    for(int i = 0; i < num_threads; ++i) {
        if(best_split == nullptr ||
           (thread_best_splits[i] != nullptr &&
            thread_best_splits[i]->gain > best_split->gain)) {
               best_split = thread_best_splits[i];
           }
        else {
            if(thread_best_splits[i] != nullptr) {
                delete thread_best_splits[i];
            }
        }
    }
    
    return best_split;
}

SplitInfo* LinearNode::FindBestSplit(DataPartition *data_partition, BoosterConfig *booster_config, int cur_class) {
    double start = omp_get_wtime();
    PrepareBinGradients(data_partition, booster_config, cur_class);
    double end = omp_get_wtime();
    booster_config->prepare_bin_time += (end - start);
    booster_config->all_prepare_bin_time += (end - start);
    
    start = omp_get_wtime();
    PrepareHistograms(data_partition, booster_config, cur_class);
    end = omp_get_wtime();
    booster_config->prepare_histogram_time += (end - start);
    booster_config->all_prepare_histogram_time += (end - start);
    
    start = omp_get_wtime();
    SplitInfo* split = FindBestSplits(data_partition, booster_config, cur_class);
    end = omp_get_wtime();
    booster_config->find_split_time += (end - start);
    booster_config->all_find_split_time += (end - start);
    
    return split;
}

void LinearNode::FillInTreePredictor(vector<int> &node_split_feature_index,
                                     vector<vector<int> > &node_linear_features,
                                     vector<vector<double> > &leaf_ks, vector<double> &leaf_bs,
                                     vector<double> &split_threshold_value, vector<uint8_t> &split_threshold_bin,
                                     vector<int> &left_childs,
                                     vector<int> &right_childs) {
    node_split_feature_index[leaf_id] = split_index;
    split_threshold_value[leaf_id] = split_threshold;
    split_threshold_bin[leaf_id] = split_bin;
    if(left_child == nullptr && right_child == nullptr) {
        leaf_ks[leaf_id] = leaf_k;
        leaf_bs[leaf_id] = leaf_b;
        node_linear_features[leaf_id] = linear_model_features;  
    }
    if(left_child != nullptr) {
        left_child->FillInTreePredictor(node_split_feature_index,
                                        node_linear_features, leaf_ks, leaf_bs,
                                        split_threshold_value, split_threshold_bin, left_childs, right_childs);
        left_childs[leaf_id] = left_child->leaf_id;
    }
    if(right_child != nullptr) {
        right_child->FillInTreePredictor(node_split_feature_index,
                                         node_linear_features, leaf_ks, leaf_bs,
                                         split_threshold_value, split_threshold_bin,left_childs, right_childs);
        right_childs[leaf_id] = right_child->leaf_id;
    }
}

LinearNode::~LinearNode() {
    if(left_child != nullptr) {
        delete left_child;
    }
    if(right_child != nullptr) {
        delete right_child;
    }   
}
