
#include "interface.hpp"
#include "booster.hpp"
#include "booster_config.hpp"
#include <string>
#include <cassert>
#include <iostream>

using std::string;
using std::cout;
using std::endl;

int SetLinearGBMParams(LinearGBMBoosterConfig booster_config,
                       const char* key, const char* value) {
    BoosterConfig* config = (BoosterConfig*)(booster_config);
    string string_key = string(key);
    string string_value = string(value);
    if(string_key == "num_trees") {
        config->num_trees = std::atoi(value);
    }
    else if(string_key == "num_leaves") {
        config->max_leaf = std::atoi(value);
    }
    else if(string_key == "min_sum_hessians") {
        config->min_sum_hessian_in_leaf = std::atof(value);
    }
    else if(string_key == "l2_reg") {
        config->l2_reg = std::atof(value);
    }
    else if(string_key == "l1_reg") {
        config->l1_reg = std::atof(value);
    }
    else if(string_key == "objective") {
        config->loss = string_value;
    }
    else if(string_key == "learning_rate") {
        config->learning_rate = std::atof(value);
    }
    else if(string_key == "eval_metric") {
        config->eval_metric = string_value;
    }
    else if(string_key == "normalization") {
        config->normalization = string_value;
    }
    else if(string_key == "num_classes") {
        config->num_classes = std::atoi(value);
    }
    else if(string_key == "num_threads") {
        config->num_threads = std::atoi(value);
    }
    else if(string_key == "num_bins") {
        config->max_bin = std::atoi(value);
    }
    else if(string_key == "min_gain") {
        config->min_gain = std::atof(value);
    }
    else if(string_key == "max_var") {
        config->num_vars = std::atoi(value);
    }
    else if(string_key == "grow_by") {
        config->grow_by = string_value;
    }
    else if(string_key == "leaf_type") {
        config->leaf_type = string_value;
    }
    else if(string_key == "max_depth") {
        config->max_level = std::atoi(value);
    }
    else if(string_key == "verbose") {
        config->verbose = std::atoi(value);
    }
    else if(string_key == "sparse_ratio") {
        config->sparse_threshold = std::atof(value);
    }
    else if(string_key == "boosting_type") {
        config->boosting_type = string_value;
    }
    else if(string_key == "goss_alpha") {
        config->goss_alpha = std::atof(value);
    }
    else if(string_key == "goss_beta") {
        config->goss_beta = std::atof(value);           
    }
    else {
        assert(false);
    } 
    
    return 0;
}

int CreateLinearGBMBoosterConfig(LinearGBMBoosterConfig *out) { *out = new BoosterConfig(); return 0; }

int CreateLinearGBM(LinearGBMBoosterConfig booster_config,
                    LinearGBMDataMat train_data,
                    LinearGBMDataMat test_data,
                    LinearGBM *out) {
    BoosterConfig& booster_configg = *static_cast<BoosterConfig*>(booster_config);
    DataMat& train_dataa = *static_cast<DataMat*>(train_data);
    DataMat& test_dataa = *static_cast<DataMat*>(test_data);
    *out = new Booster(*static_cast<BoosterConfig*>(booster_config), *static_cast<DataMat*>(train_data), *static_cast<DataMat*>(test_data));    
    return 0;
}

int CreateLinearGBMDataMat(LinearGBMBoosterConfig booster_config, const char* name,
                           int label_index, int query_index,
                           const char* file_path, LinearGBMDataMat *out, LinearGBMDataMat reference) {

    if(reference != nullptr) { 
	    *out = new DataMat(*(BoosterConfig*)booster_config, string(name), label_index, query_index, string(file_path), (DataMat*)reference);  
    }
    else {
 	    *out = new DataMat(*(BoosterConfig*)booster_config, string(name), label_index, query_index, string(file_path));  
    }
    return 0;
}

int Train(LinearGBM gbm) {
	cout << "[GBDT-PL] start training" << endl;
    ((Booster*)gbm)->Train();
    return 0;
}

int LinearGBMPrintBoosterConfig(LinearGBMBoosterConfig booster_config) {
    ((BoosterConfig*)booster_config)->PrintAllParams();
}

int LinearGBMPredict(LinearGBM booster, LinearGBMDataMat test_data, double** preds, int *num_data, int iters) {  
    DataMat &test_dataa = *(DataMat*)(test_data);
    test_dataa.predict_values.resize(test_dataa.num_data, 0.0);
    ((Booster*)booster)->Predict(test_dataa, test_dataa.predict_values, iters);
    *preds = test_dataa.predict_values.data(); 
    *num_data = test_dataa.num_data;
}

int LinearGBMBestIteration(LinearGBM booster, int* best_iteration, double* best_score) {
    *best_iteration = ((Booster*)booster)->get_best_iteration(best_score);
}

int LinearGBMGetScoresPerIteration(LinearGBM booster, const char* name, double** scores) {
    *scores = ((Booster*)booster)->get_per_iteration_scores(string(name)).data();
}
