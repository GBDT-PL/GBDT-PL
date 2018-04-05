//
//  additive_linear_node.cpp
//  LinearGBMVector
//


//

#include "node.hpp"
#include <mkl.h>

AdditiveLinearNode::AdditiveLinearNode(DataPartition *data_partition):
LinearNode(data_partition) {
    parent = nullptr;
    left_child = nullptr;
    right_child = nullptr;      
    sibling = nullptr; 
}

AdditiveLinearNode::AdditiveLinearNode(const vector<double> &ks, const double &b,   
                       vector<int> linear_model_featuress, int leaf_idd, bool smaller_nodee,
                       vector<RowHistogram*>* histogramm,
                                       AdditiveLinearNode* parentt, double gainn):
LinearNode(ks, b, linear_model_featuress, leaf_idd, smaller_nodee, histogramm, parentt, gainn) {
    parent = parentt;
    left_child = nullptr;
    right_child = nullptr;
    sibling = nullptr;
}

void AdditiveLinearNode::PrepareBinGradients(DataPartition *data_partition,
                                             BoosterConfig *booster_config, int cur_class) {
    int leaf_start = data_partition->get_leaf_starts()[leaf_id];
    int leaf_end = data_partition->get_leaf_ends()[leaf_id];
    float* local_gradients = data_partition->get_all_gradients(cur_class);
    const fvec64& pred_values = data_partition->get_pred_values();
    int leaf_num_data = leaf_end - leaf_start;
    
    const int *local_data_indices = data_partition->get_data_indices().data() + leaf_start;         
    
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
        if(bin_gradients.size() < 5 * leaf_num_data) {
            bin_gradients.resize(5 * leaf_num_data);
        }
        
        float *bin_gradients_ptr = bin_gradients.data();
#pragma omp parallel for schedule(static) num_threads(booster_config->num_threads)
        for(int i = 0; i < leaf_end - leaf_start; ++i) {    
            float *local_bin_gradient = bin_gradients_ptr + i * 5;
            int index = local_data_indices[i];
            const float gradient = local_gradients[2 * index];
            const float hessian = local_gradients[2 * index + 1];
            const float pred_value = pred_values[index];
            local_bin_gradient[0] = gradient;
            local_bin_gradient[1] = hessian;
            local_bin_gradient[2] = gradient * pred_value;  
            local_bin_gradient[3] = hessian * pred_value;
            local_bin_gradient[4] = hessian * pred_value * pred_value;
        }
    }
    else if(parent->get_feature_num() < linear_model_features.size()) {
        if(bin_gradients.size() < 3 * leaf_num_data) {
            bin_gradients.resize(3 * leaf_num_data, 0.0);
        }
        
        float *bin_gradients_ptr = bin_gradients.data();
#pragma omp parallel for schedule(static) num_threads(booster_config->num_threads)
        for(int i = 0; i < leaf_end - leaf_start; ++i) {
            float *local_bin_gradient = bin_gradients_ptr + i * 3;
            int index = local_data_indices[i];
            const float gradient = local_gradients[2 * index];
            const float hessian = local_gradients[2 * index + 1];
            const float pred_value = pred_values[index]; 
            local_bin_gradient[0] = gradient * pred_value;
            local_bin_gradient[1] = hessian * pred_value;
            local_bin_gradient[2] = hessian * pred_value * pred_value;                                      
        }
    }
}

void AdditiveLinearNode::PrepareHistograms(DataPartition *data_partition,
                                           BoosterConfig *booster_config, int cur_class) {
    const vector<int>& useful_features = data_partition->get_useful_features();
    const vector<OrderedFeature*> &features = data_partition->get_features();
    fvec64 &bin_gradients = data_partition->get_bin_gradients();
    bool need_augment = true;
    if(parent != nullptr && parent->get_feature_num() == linear_model_features.size()) {
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
            sibling_hist = (*(sibling->get_histogram()))[fid];      
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
                cur_num_vars = 2;
                prev_num_vars = 1;
            }
            else {
                cur_num_vars = 1;
                prev_num_vars = 1;
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

SplitInfo* AdditiveLinearNode::FindBestSplits(DataPartition *data_partition,
                                       BoosterConfig *booster_config, int cur_class) {
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

void AdditiveLinearNode::Split(SplitInfo *split, AdditiveLinearNode **out_left,
                               AdditiveLinearNode **out_right, int max_var,
                               DataPartition *data_partition, int num_threads) {
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
    redundant |= (linear_model_features.size() == max_var);
    if(!redundant) {
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
        left_child = new AdditiveLinearNode(best_split->left_ks, best_split->left_b, to_left,
                                    best_split->left_leaf_id, left_smaller,
                                    nullptr, this, best_split->left_gain);
        right_child = new AdditiveLinearNode(best_split->right_ks, best_split->right_b, to_right,
                                     best_split->right_leaf_id, !left_smaller,
                                     histogram, this, best_split->right_gain);
    }
    else {
        left_child = new AdditiveLinearNode(best_split->left_ks, best_split->left_b, to_left,
                                    best_split->left_leaf_id, left_smaller,
                                    histogram, this, best_split->left_gain);
        right_child = new AdditiveLinearNode(best_split->right_ks, best_split->right_b, to_right,       
                                     best_split->right_leaf_id, !left_smaller,
                                     nullptr, this, best_split->right_gain);
    }
    left_child->set_sibling(right_child);
    right_child->set_sibling(left_child);
    *out_left = left_child;
    *out_right = right_child;
        
    if(!redundant) {
        const uint8_t* bins = data_partition->get_features()[best_split->feature_index]->get_bins();
        const double* values = data_partition->get_feature_values(best_split->feature_index);
        
        fvec64 &pred_values = data_partition->get_pred_values();
        
        if(leaf_id == 0) {
            double left_k = best_split->left_ks.back();
            double right_k = best_split->right_ks.back();
            
            data_partition->reset_pred_values();
            const int* data_indices_left = data_partition->get_data_indices().data() + left_leaf_start;
#pragma omp parallel for schedule(static) num_threads(num_threads)
            for(int i = 0; i < left_leaf_end - left_leaf_start; ++i) {
                int index = data_indices_left[i];
                pred_values[index] = left_k * values[bins[index]];
            }
            const int* data_indices_right = data_partition->get_data_indices().data() + right_leaf_start;
#pragma omp parallel for schedule(static) num_threads(num_threads)
            for(int i = 0; i < right_leaf_end - right_leaf_start; ++i) {
                int index = data_indices_right[i];
                pred_values[index] = right_k * values[bins[index]];
            }
        }
        else {
            double left_k1 = best_split->left_ks[0];
            double left_k2 = best_split->left_ks[1];
            double right_k1 = best_split->right_ks[0];
            double right_k2 = best_split->right_ks[1];
            
            const int* data_indices_left = data_partition->get_data_indices().data() + left_leaf_start;
#pragma omp parallel for schedule(static) num_threads(num_threads)
            for(int i = 0; i < left_leaf_end - left_leaf_start; ++i) {
                int index = data_indices_left[i];
                pred_values[index] = left_k1 * pred_values[index] + left_k2 * values[bins[index]];
            }
            const int* data_indices_right = data_partition->get_data_indices().data() + right_leaf_start;
#pragma omp parallel for schedule(static) num_threads(num_threads)
            for(int i = 0; i < right_leaf_end - right_leaf_start; ++i) {
                int index = data_indices_right[i];
                pred_values[index] = right_k1 * pred_values[index] +  right_k2 * values[bins[index]];
            }
        }
    }
}

void AdditiveLinearNode::FillInTreePredictor(vector<int> &node_split_feature_index,
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

void AdditiveLinearNode::AfterTrain(DataPartition *data_partition, int cur_class,
                                    BoosterConfig *booster_config) {  
    if(left_child != nullptr && right_child != nullptr) {
        return;
    }
    else {
        const vector<OrderedFeature*>& features = data_partition->get_features();
        int num_vars = static_cast<int>(linear_model_features.size());
        double l2_reg = booster_config->l2_reg;
        vector<double> matrix((num_vars + 1) * (num_vars + 1), 0.0);
        vector<double> vec(num_vars + 1, 0.0);
        vector<const uint8_t*> fbins;
        vector<double*> fvalues;
        
        int leaf_start = data_partition->get_leaf_starts()[leaf_id];
        int leaf_end = data_partition->get_leaf_ends()[leaf_id];
        
        for(int i = 0; i < num_vars; ++i) {
            fbins.push_back(features[linear_model_features[i]]->GetLocalData(leaf_id, leaf_start));
            fvalues.push_back(features[linear_model_features[i]]->get_feature_values());               
        }
        
        const float* gradients = data_partition->get_all_gradients(cur_class);
        const int* local_data_indices = data_partition->get_data_indices().data() + leaf_start;
        for(int k = 0; k < leaf_end - leaf_start; ++k) {
            int index = local_data_indices[k];
            double gradient = gradients[2 * index];
            double hessian = gradients[2 * index + 1];  
            for(int i = 0; i < num_vars; ++i) {
                double value1 = fvalues[i][fbins[i][k]];
                matrix[i * (num_vars + 1) + i] += value1 * value1 * hessian;
                matrix[i * (num_vars + 1) + num_vars] += value1 * hessian;
                vec[i] -= value1 * gradient;
                for(int j = i + 1; j < num_vars; ++j) {
                    double value2 = fvalues[j][fbins[j][k]];
                    matrix[i * (num_vars + 1) + j] += value1 * value2 * hessian;
                }
            }
            matrix.back() += hessian;
            
            vec.back() -= gradient;
        }
        
        matrix[0] += l2_reg;
        for(int k = 1; k < num_vars + 1; ++k) {
            matrix[k * (num_vars + 1) + k] += l2_reg;
            for(int s = 0; s < k; ++s) {
                matrix[k * (num_vars + 1) + s] = matrix[s * (num_vars + 1) + k];
            }
        }
        
        int mkl_n = num_vars + 1, mkl_m = 1, mkl_info = 1;
        int *mkl_ipiv = new int[mkl_n];
        dgesv(&mkl_n, &mkl_m, matrix.data(), &mkl_n, mkl_ipiv, vec.data(), &mkl_n, &mkl_info);
        
        leaf_k.resize(num_vars, 0.0);
#pragma omp simd
        for(int i = 0; i < num_vars; ++i) {
            leaf_k[i] = vec[i];
        }
        leaf_b = vec.back();
        delete [] mkl_ipiv; 
    }
}

void AdditiveLinearNode::UpdateTrainScore(const vector<int> &indices, double *scores,
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
        
        int leaf_start = leaf_starts[leaf_id], leaf_end = leaf_ends[leaf_id];
        const int *local_data_indices = indices.data() + leaf_start;
        
        for(int j = 0; j < leaf_num_features; ++j) {
            data_bins[j] = (static_cast<LinearBinFeature*>(features[linear_model_features[j]]))->GetLocalData(leaf_id, leaf_start);
            bin_valuess[j] = features[linear_model_features[j]]->get_feature_values();
        }
        
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

void AdditiveLinearNode::GetAllLeaves(vector<Node *> &leaves) { 
    if(left_child == nullptr && right_child == nullptr) {
        leaves.push_back(this);
    }
    else {
        left_child->GetAllLeaves(leaves);
        right_child->GetAllLeaves(leaves);
    }
}
