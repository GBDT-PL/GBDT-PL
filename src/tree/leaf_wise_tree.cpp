//
//  leaf_wise_tree.cpp
//  LinearGBM
//


//

#include "tree.hpp"
#include <assert.h>
#include <omp.h>

/*
 Constructor.
 @param _data_mat: Training data from the booster.
 @param _booster_config: Configurations of the booster.
 */
template <typename NODE_TYPE>
LeafWiseTree<NODE_TYPE>::LeafWiseTree(DataPartition* _data_partition,
                                      BoosterConfig *_booster_config, int _cur_class):
Tree(_data_partition, _booster_config, _cur_class) {
    nodes.resize(2 * booster_config->max_leaf + 1, nullptr);
}

template <typename NODE_TYPE>
void LeafWiseTree<NODE_TYPE>::Init() {
    root = new NODE_TYPE(data_partition);   
    nodes[0] = root;
}

/*
 Train a decision tree, using leaf-wise growing.
 @param tree_predictor: output parameter for the tree predictor.
 */
template <typename NODE_TYPE>
void LeafWiseTree<NODE_TYPE>::Train() {
    //indices of the two newly grown leaves, -1 for no leaf.
    int left_leaf_id = 0, right_leaf_id = -1;
    
    int cnt = 0;
    
    //Train the tree iteratively, each time pick the leaf with maximum gain to split.
    while(cur_leaf_num < 2 * booster_config->max_leaf + 1) {
        
        FindBestSplitForTwoLeaves(left_leaf_id, right_leaf_id);
        
        double max_gain = booster_config->min_gain;
        int best_split_leaf = -1;
        
        ++cnt;
        
        for(int i = 0; i < cur_leaf_num; ++i) {
            if(best_splits[i] != nullptr && best_splits[i]->gain > max_gain) {  
                max_gain = best_splits[i]->gain;
                best_split_leaf = i;
            }
        }
        
        if(best_split_leaf != -1) {
            //std::cout << cnt << " best gain: " << (best_splits[best_split_leaf]->gain) << std::endl;
            
            left_leaf_id = cur_leaf_num;
            right_leaf_id = cur_leaf_num + 1;
            
            best_splits[best_split_leaf]->SetChildrenID(left_leaf_id, right_leaf_id);
            
            cur_leaf_num += 2;
            double start = omp_get_wtime();
            data_partition->Split(best_split_leaf, best_splits[best_split_leaf], cur_class);
            
            nodes[best_split_leaf]->Split(best_splits[best_split_leaf], 
                                          &nodes[left_leaf_id],
                                          &nodes[right_leaf_id],
                                          booster_config->num_vars,
                                          data_partition,
                                          booster_config->num_threads);
            
            double end = omp_get_wtime();
            booster_config->split_time += (end - start);
            booster_config->all_split_time += (end - start); 
            
            assert(best_splits[best_split_leaf] != nullptr);
            best_splits[best_split_leaf]->gain = booster_config->min_gain;
        }
        else {
            for(int i = 0; i < best_splits.size(); ++i) {
                if(best_splits[i] != nullptr) {
                    delete best_splits[i];
                    best_splits[i] = nullptr;
                }
            }
            return;
        }
    }
    
    for(int i = 0; i < best_splits.size(); ++i) {
        if(best_splits[i] != nullptr) {
            delete best_splits[i];
            best_splits[i] = nullptr;               
        }
    }
    best_splits.clear();
    best_splits.shrink_to_fit();
}


// Get the best splits for two newly grown leaves.
template <typename NODE_TYPE>
void LeafWiseTree<NODE_TYPE>::FindBestSplitForTwoLeaves(int left_leaf_id, int right_leaf_id) {
    if(right_leaf_id == -1) {
        SplitInfo *left_best_split = nullptr;
        left_best_split = nodes[left_leaf_id]->FindBestSplit(data_partition, booster_config, cur_class);
        if(left_leaf_id == best_splits.size()) {
            best_splits.push_back(left_best_split);
        }
        else {
            best_splits[left_leaf_id] = left_best_split;    
        }
        return;
    }
    
    int left_leaf_size = data_partition->GetLeafSize(left_leaf_id);
    int right_leaf_size = data_partition->GetLeafSize(right_leaf_id);
    if(left_leaf_size <= right_leaf_size) {
        SplitInfo *left_best_split = nullptr, *right_best_split = nullptr;
        if(left_leaf_id != -1) {
            left_best_split = nodes[left_leaf_id]->FindBestSplit(data_partition, booster_config, cur_class);
            if(left_leaf_id == best_splits.size()) {
                best_splits.push_back(left_best_split);
            }
            else {
                best_splits[left_leaf_id] = left_best_split;
            }
        }
        
        if(right_leaf_id != -1) {
            right_best_split = nodes[right_leaf_id]->FindBestSplit(data_partition, booster_config, cur_class);
            if(right_leaf_id == best_splits.size()) {
                best_splits.push_back(right_best_split);            
            }
            else {
                best_splits[right_leaf_id] = right_best_split;
            }
        }
    }
    else {
        SplitInfo *left_best_split = nullptr, *right_best_split = nullptr;  
        
        if(right_leaf_id != -1) {
            right_best_split = nodes[right_leaf_id]->FindBestSplit(data_partition, booster_config, cur_class);
        }
        if(left_leaf_id != -1) {
            left_best_split = nodes[left_leaf_id]->FindBestSplit(data_partition, booster_config, cur_class);
        }
        
        if(left_leaf_id != -1) {
            if(left_leaf_id == best_splits.size()) {
                best_splits.push_back(left_best_split);
            }
            else {
                best_splits[left_leaf_id] = left_best_split;
            }
        }
        
        if(right_leaf_id != -1) {
            if(right_leaf_id == best_splits.size()) {
                best_splits.push_back(right_best_split);
            }
            else {
                best_splits[right_leaf_id] = right_best_split;  
            }
        }
    }
}

template <typename NODE_TYPE>
void LeafWiseTree<NODE_TYPE>::UpdateTrainScore(double *scores) {
    vector<Node*> leaves;
    root->GetAllLeaves(leaves);
//#pragma omp parallel for schedule(static) num_threads(booster_config->num_threads)
    for(int i = 0; i < leaves.size(); ++i) {
        leaves[i]->UpdateTrainScore(data_partition->get_data_indices(),                 
                           scores,
                           data_partition->get_features(),  
                           booster_config,
                           data_partition->get_leaf_starts(),
                           data_partition->get_leaf_ends());
    }
}

template <typename NODE_TYPE>
void LeafWiseTree<NODE_TYPE>::AfterTrain(DataPartition *data_partition, int cur_class,
                                         BoosterConfig *booster_config) {
    vector<Node*> leaves;
    root->GetAllLeaves(leaves);
#pragma omp parallel for schedule(static) num_threads(booster_config->num_threads)
    for(int i = 0; i < leaves.size(); ++i) {
        leaves[i]->AfterTrain(data_partition, cur_class, booster_config); 
    }
}
