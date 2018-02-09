//
//  level_wise_tree.cpp
//  LinearGBM
//


//

#include "tree.hpp"

template <typename NODE_TYPE>
LevelWiseTree<NODE_TYPE>::LevelWiseTree(DataPartition* _data_partition, BoosterConfig* _booster_config, int _cur_class):
Tree(_data_partition, _booster_config, _cur_class) {
    best_splits.resize((1 << (booster_config->max_level + 1)) - 1, nullptr);    
    is_leaf.resize((1 << (booster_config->max_level + 1)) - 1, false);
    nodes.resize((1 << (booster_config->max_leaf + 1)) - 1, nullptr);
    root = new NODE_TYPE(_data_partition);
    nodes[0] = root;
}

template <typename NODE_TYPE>
void LevelWiseTree<NODE_TYPE>::TrainLevel(int level) {
    if(level == 1) {
        best_splits[0] = root->FindBestSplit(data_partition, booster_config, cur_class);
        
        is_leaf[0] = false;
        if(best_splits[0] != nullptr) {
            std::cout << 0 << " best gain " << best_splits[0]->gain << std::endl;
            best_splits[0]->SetChildrenID(1, 2);
            is_leaf[1] = true;
            is_leaf[2] = true;
            data_partition->Split(0, best_splits[0], cur_class);    
        }
        return;
    }
    
    for(int leaf_id = (1 << (level - 2)) - 1; leaf_id < (1 << (level - 1)) - 1; ++leaf_id) {
        if((is_leaf[2 * leaf_id + 1] && is_leaf[2 * leaf_id + 2] &&
            data_partition->GetLeafSize(2 * leaf_id + 1) <= data_partition->GetLeafSize(2 * leaf_id + 2))) {
            best_splits[2 * leaf_id + 1] = nodes[2 * leaf_id + 1]->FindBestSplit(data_partition, booster_config, cur_class);
            is_leaf[2 * leaf_id + 1] = false;
            if(best_splits[2 * leaf_id + 1] != nullptr) {
                std::cout << (2 * leaf_id + 1) << " best gain " << best_splits[2 * leaf_id + 1]->gain << std::endl;
                best_splits[2 * leaf_id + 1]->SetChildrenID(4 * leaf_id + 3, 4 * leaf_id + 4);
                is_leaf[4 * leaf_id + 3] = true;
                is_leaf[4 * leaf_id + 4] = true;
                data_partition->Split(2 * leaf_id + 1, best_splits[2 * leaf_id + 1], cur_class);
            }
            
            best_splits[2 * leaf_id + 2] = nodes[2 * leaf_id + 2]->FindBestSplit(data_partition, booster_config, cur_class);
            is_leaf[2 * leaf_id + 2] = false;
            if(best_splits[2 * leaf_id + 2] != nullptr) {
                std::cout << (2 * leaf_id + 2) << " best gain " << best_splits[2 * leaf_id + 2]->gain << std::endl;
                best_splits[2 * leaf_id + 2]->SetChildrenID(4 * leaf_id + 5, 4 * leaf_id + 6);
                is_leaf[4 * leaf_id + 5] = true;
                is_leaf[4 * leaf_id + 6] = true;
                data_partition->Split(2 * leaf_id + 2, best_splits[2 * leaf_id + 2], cur_class);
            }
        }
        else if(is_leaf[2 * leaf_id + 1] && is_leaf[2 * leaf_id + 2] &&
                data_partition->GetLeafSize(2 * leaf_id + 1) > data_partition->GetLeafSize(2 * leaf_id + 2)) {
            best_splits[2 * leaf_id + 2] = nodes[2 * leaf_id + 2]->FindBestSplit(data_partition, booster_config, cur_class);
            is_leaf[2 * leaf_id + 2] = false;
            if(best_splits[2 * leaf_id + 2] != nullptr) {
                std::cout << (2 * leaf_id + 2) << " best gain " << best_splits[2 * leaf_id + 2]->gain << std::endl;
                best_splits[2 * leaf_id + 2]->SetChildrenID(4 * leaf_id + 5, 4 * leaf_id + 6);
                is_leaf[4 * leaf_id + 5] = true;
                is_leaf[4 * leaf_id + 6] = true;
                data_partition->Split(2 * leaf_id + 2, best_splits[2 * leaf_id + 2], cur_class);
            }
            best_splits[2 * leaf_id + 1] = nodes[2 * leaf_id + 1]->FindBestSplit(data_partition, booster_config, cur_class);
            is_leaf[2 * leaf_id + 1] = false;
            if(best_splits[2 * leaf_id + 1] != nullptr) {
                std::cout << (2 * leaf_id + 1) << " best gain " << best_splits[2 * leaf_id + 1]->gain << std::endl;
                best_splits[2 * leaf_id + 1]->SetChildrenID(4 * leaf_id + 3, 4 * leaf_id + 4);
                is_leaf[4 * leaf_id + 3] = true;
                is_leaf[4 * leaf_id + 4] = true;
                data_partition->Split(2 * leaf_id + 1, best_splits[2 * leaf_id + 1], cur_class);
            }
        }
    }
}

template <typename NODE_TYPE>
void LevelWiseTree<NODE_TYPE>::Train() {
    is_leaf[0] = true;
    for(int i = 0; i < booster_config->max_level; ++i) {
        TrainLevel(i + 1);
    }
    for(int i = 0; i < best_splits.size(); ++i) {
        if(best_splits[i] != nullptr) {
            delete best_splits[i];
            best_splits[i] = nullptr;
        }
    }
}


template <typename NODE_TYPE>
void LevelWiseTree<NODE_TYPE>::UpdateTrainScore(double *scores) {   
    
}


template <typename NODE_TYPE>
void LevelWiseTree<NODE_TYPE>::ShrinkToPredictor() {} 
