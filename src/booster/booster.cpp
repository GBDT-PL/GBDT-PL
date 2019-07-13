//
//  booster.cpp
//  LinearGBM
//


//

#include "booster.hpp"
#include <cmath>
#include <string>
#include <time.h>
#include <cstdlib>
#include <omp.h>
#include <fstream>
#include <vector>
#include <functional>
#include "../tree/leaf_wise_tree.cpp"
#include "../tree/level_wise_tree.cpp"
#include "argmax.h"
#include "../argmax/argmax.cpp"
#include "threshold_finder.hpp" 

using std::vector;
using std::cout;
using std::endl;

Booster::Booster(BoosterConfig& _booster_config,
                 DataMat& _train_data,
                 DataMat& _test_data):
booster_config(_booster_config), train_data(_train_data), test_data(_test_data),
data_partition(&_train_data, &_booster_config, gradients) {
    num_threads = booster_config.num_threads;
    
    int num_classes = booster_config.num_classes;
    int num_trees = booster_config.num_trees;

    avg_label = 0.0;
    
    data_indices_tmp.resize(train_data.num_data, 0);
    
    train_predict_values.resize(train_data.num_data * booster_config.num_classes, 0.0);
    test_predict_values.resize(test_data.num_data * booster_config.num_classes, 0.0);
    
    if(train_data.num_data % 4 == 0) {
        num_data_aligned = train_data.num_data;
    }
    else {
        num_data_aligned = (train_data.num_data / 4 + 1) * 4;
    }
    gradients.resize(num_data_aligned * 2 * booster_config.num_classes, 0.0);
   
    
    trees.resize(num_trees * num_classes, nullptr); 
    
    srand((int)time(NULL));
    
    evaluate_time = 0.0;
    best_iteration = -1;
    scores_per_iteration.emplace(train_data.name, vector<double>(0));
    scores_per_iteration.emplace(test_data.name, vector<double>(0));
    scores_per_iteration[train_data.name].resize(booster_config.num_trees, 0.0);
    scores_per_iteration[test_data.name].resize(booster_config.num_trees, 0.0);
}

void Booster::SetupTrees() {
    if(booster_config.grow_by == "leaf") {
        if(booster_config.leaf_type == "half_additive") {
            for(int i = 0; i < booster_config.num_trees; ++i) {
                for(int j = 0; j < booster_config.num_classes; ++j) {
                    trees[i * booster_config.num_classes + j] =
                        new LeafWiseTree<AdditiveLinearNode>(&data_partition, &booster_config, j); 
                }
            }
        }
        else if(booster_config.leaf_type == "constant") {
            for(int i = 0; i < booster_config.num_trees; ++i) {
                for(int j = 0; j < booster_config.num_classes; ++j) {
                    trees[i * booster_config.num_classes + j] =
                    new LeafWiseTree<ConstantNode>(&data_partition, &booster_config, j);    
                }
            }
        }
        else if(booster_config.leaf_type == "linear") {
            for(int i = 0; i < booster_config.num_trees; ++i) {
                for(int j = 0; j < booster_config.num_classes; ++j) {
                    trees[i * booster_config.num_classes + j] =
                    new LeafWiseTree<LinearNode>(&data_partition, &booster_config, j);
                }
            }
        }
    }
    else if(booster_config.grow_by == "level") {
        if(booster_config.leaf_type == "half_additive") {
            for(int i = 0; i < booster_config.num_trees; ++i) {
                for(int j = 0; j < booster_config.num_classes; ++j) {
                    trees[i * booster_config.num_classes + j] =
                    new LevelWiseTree<AdditiveLinearNode>(&data_partition, &booster_config, j);
                }
            }
        }
        else if(booster_config.leaf_type == "constant") {
            for(int i = 0; i < booster_config.num_trees; ++i) {
                for(int j = 0; j < booster_config.num_classes; ++j) {
                    trees[i * booster_config.num_classes + j] =
                    new LevelWiseTree<ConstantNode>(&data_partition, &booster_config, j);               
                }
            }
        }
        else if(booster_config.leaf_type == "linear") {
            for(int i = 0; i < booster_config.num_trees; ++i) {
                for(int j = 0; j < booster_config.num_classes; ++j) {
                    trees[i * booster_config.num_classes + j] =
                    new LevelWiseTree<LinearNode>(&data_partition, &booster_config, j);
                }
            }
        }
    }
}

void Booster::GetMask(double prob, std::vector<bool> &mask) {
#pragma omp parallel for schedule(static) num_threads(num_threads)          
    for(int i = 0; i < mask.size(); ++i) {
        if(rand() * 1.0 / RAND_MAX <= prob) {
            mask[i] = true;
        }
        else {
            mask[i] = false;    
        }
    }
}

void Booster::SetupEvals() {
    if(booster_config.verbose > 0) {
        if(booster_config.eval_metric == "auc") {
	    if(booster_config.verbose == 2) {
            train_eval = new AUC(booster_config, train_data.label, train_predict_values);
	    }
            test_eval = new AUC(booster_config, test_data.label, test_predict_values);
        }
        else if(booster_config.eval_metric == "rmse") {
	    if(booster_config.verbose == 2) {
            train_eval = new RMSE(booster_config, train_data.label, train_predict_values);
	    }
            test_eval = new RMSE(booster_config, test_data.label, test_predict_values);
        }
        else if(booster_config.eval_metric == "ndcg") {
	    if(booster_config.verbose == 2) {
            train_eval = new NDCG(booster_config, train_data.label, train_predict_values, train_data.queries);
	    }
            test_eval = new NDCG(booster_config, test_data.label, test_predict_values, test_data.queries);
        }
        else if(booster_config.eval_metric == "acc") {
	    if(booster_config.verbose == 2) {
            train_eval = new ACC(booster_config, train_data.label, train_predict_values);
	    }
            test_eval = new ACC(booster_config, test_data.label, test_predict_values);
        } 
    }
}

void Booster::SetupObjs() {
    if(booster_config.loss == "logistic") {
        objective = new Binary(gradients, train_data.label,
                               train_predict_values, booster_config, train_data.num_data);
    }
    else if(booster_config.loss == "l2") {
        objective = new Regression(gradients, train_data.label,
                                   train_predict_values, booster_config, train_data.num_data);
    }
    else if(booster_config.loss == "multi-logistic") {
        objective = new MultiClass(gradients, train_data.label,
                                   train_predict_values, booster_config, train_data.num_data, num_data_aligned);
    }
}

inline float RandFloat(int seed) {
    thread_local static int x = seed;
    x = (214013 * x + 2531011);
    return static_cast<float>(static_cast<int>((x >> 16) & 0x7FFF) / 32768.0f);
}

void Booster::Sample(int iteration, int cur_class) {
    
    struct GradientIndex {
        double gradient;
        int index;
        GradientIndex(double gradientt, int indexx) {
            gradient = gradientt;
            index = indexx;
        }
    };
    
    std::srand(0);
    
    if(booster_config.boosting_type == "goss" && iteration >= 1.0 / booster_config.learning_rate) {     
        vector<int> thread_offsets(num_threads + 1, 0);
        vector<double> thread_threshold(num_threads, 0.0);
        
        int top_k = booster_config.goss_alpha * train_data.num_data;    
        int offset = cur_class * num_data_aligned;
        
        double amplify = (1 - booster_config.goss_alpha) / booster_config.goss_beta;
        int chunk_size = (train_data.num_data + num_threads - 1) / num_threads;
        
        float threshold = ThresholdFinder(booster_config.num_threads).
            FindThreshold(gradients.data() + 2 * offset, top_k, 255, train_data.num_data);
        
#pragma omp parallel for schedule(static, 1) num_threads(booster_config.num_threads)
        for(int i = 0; i < booster_config.num_threads; ++i) {
            int thread_start = i * chunk_size;
            int thread_end = std::min(thread_start + chunk_size, train_data.num_data);
            
            int cnt = 0;
            int *data_indices_tmp_ptr = data_indices_tmp.data() + thread_start;
            for(int j = thread_start; j < thread_end; ++j) {
                double abs = std::fabs(gradients[(j + offset) * 2] * gradients[(j + offset) * 2 + 1]);
                if(abs >= std::abs(threshold)) {
                    data_indices_tmp_ptr[cnt++] = j;    
                }
                else {
                    float rand_float = RandFloat(iteration * num_threads + i);
                    if(rand_float <= booster_config.goss_beta / (1 - booster_config.goss_alpha)) {
                        data_indices_tmp_ptr[cnt++] = j;
                        gradients[(j + offset) * 2] *= amplify;
                        gradients[((j + offset) * 2) + 1] *= amplify;           
                    }
                }
            }
            thread_offsets[i + 1] = cnt;
        }
        
        for(int i = 0; i < num_threads; ++i) {
            thread_offsets[i + 1] += thread_offsets[i];
        }
        
        vector<int> &data_indices = data_partition.get_data_indices();
        
#pragma omp parallel for schedule(static, 1) num_threads(booster_config.num_threads)
        for(int i = 0; i < num_threads; ++i) {
            memcpy(data_indices.data() + thread_offsets[i],
                   data_indices_tmp.data() + i * chunk_size,
                   sizeof(int) * (thread_offsets[i + 1] - thread_offsets[i]));
        }
        
        data_partition.Sample(nullptr, thread_offsets.back());
    }
}

void Booster::Train() {
    double all_start_t = omp_get_wtime();
    
    data_partition.BeforeTrain();   
    train_data.Shrink();
    test_data.Shrink();             
    SetupTrees();
    SetupEvals();
    SetupObjs();
    if(booster_config.loss == "l2") {
        BoostFromAverage();
    }
    for(int i = 0; i < booster_config.num_trees; ++i) {
        double start_t = omp_get_wtime();
        for(int j = 0; j < booster_config.num_classes; ++j) {               
            data_partition.BeforeTrainTree(j, i);
            double start = omp_get_wtime();
            objective->UpdateGradients(j);
            Sample(i, j);
            double end = omp_get_wtime();
            booster_config.update_gradients_time = (end - start);
            booster_config.all_update_gradients_time += (end - start);
            int train_offset = j * train_data.num_data;
            int test_offset = j * test_data.num_data;
            trees[i * booster_config.num_classes + j]->Init();  
            trees[i * booster_config.num_classes + j]->Train();
            start = omp_get_wtime();
            trees[i * booster_config.num_classes + j]->AfterTrain(&data_partition,
                                                                  j, &booster_config);
            end = omp_get_wtime();
            booster_config.after_train_tree_time = end - start;
            booster_config.all_after_train_tree_time += (end - start); 
            
            if(booster_config.boosting_type != "goss") {
                start = omp_get_wtime();
                trees[i * booster_config.num_classes + j]->UpdateTrainScore(train_predict_values.data() + train_offset);
                end = omp_get_wtime();
                booster_config.update_train_score_time = (end - start);
                booster_config.all_update_train_score_time += (end - start);
            }
            
            if(booster_config.verbose > 0) {
                trees[i * booster_config.num_classes + j]->ShrinkToPredictor(); 
                trees[i * booster_config.num_classes + j]->Predict(&test_data,
                                                                   test_predict_values.data() + test_offset);
            }
            if(booster_config.boosting_type == "goss") {
                start = omp_get_wtime();
                trees[i * booster_config.num_classes + j]->ShrinkToPredictor();
                trees[i * booster_config.num_classes + j]->PredictTrain(&data_partition, &train_data, 
                                                                   train_predict_values.data() + train_offset);
                end = omp_get_wtime();
                booster_config.update_train_score_time = (end - start);
                booster_config.all_update_train_score_time += (end - start);        
            }
            data_partition.AfterTrainTree(); 
        }
        cout << "[GBDT-PL] iteration " << i << endl;
        if(booster_config.verbose > 0) {
            if(booster_config.verbose == 2) {
                double train_eval_score = train_eval->Eval();
                scores_per_iteration[train_data.name][i] = train_eval_score;
                cout << "[GBDT-PL] " << train_data.name << " " << train_eval->name << ": " << train_eval_score << endl;
            }
            double test_eval_score = test_eval->Eval();
            scores_per_iteration[test_data.name][i] = test_eval_score;
            cout << "[GBDT-PL] " << test_data.name << " " << test_eval->name << ": " << test_eval_score << endl;
        }
        //booster_config.PrintIterationTime();
        booster_config.ClearIterationTime(); 
        std::cout << "[GBDT-PL] time: " << (omp_get_wtime() - start_t) << std::endl;
	cout << endl;
    }
    std::cout << "[GBDT_PL] all time: " << (omp_get_wtime() - all_start_t) << std::endl;
    double best_test_score = scores_per_iteration[test_data.name][0];
    best_iteration = 0;
    std::function<bool(double,double)> comp_func;
    if(test_eval->larger_is_better) {
        comp_func = [](double score1, double score2) { return score1 <= score2; };
    }
    else {
        comp_func = [](double score1, double score2) { return  score1 >= score2; };
    }
    for(int i = 1; i < booster_config.num_trees; ++i) {
        if(comp_func(best_test_score, scores_per_iteration[test_data.name][i] )) {
            best_test_score = scores_per_iteration[test_data.name][i];
            best_iteration = i;
        }
    }
}

int Booster::get_best_iteration(double* best_score) { 
    *best_score = scores_per_iteration[test_data.name][best_iteration];
    return best_iteration; 
}

vector<double>& Booster::get_per_iteration_scores(string data_name) {
    if(data_name == train_data.name && booster_config.verbose != 2) {
        cout << "[GBDT-PL]: please set verbose = 2 when training to get the per iteration train score. Otherwise zero scores will be returned." << endl;
    }
    if(data_name == test_data.name && booster_config.verbose == 0) {
        cout << "[GBDT-PL]: please set verbose = 1 or 2 when training to get the per iteration evaluation score. Otherwise zero scores will be returned." << endl;
    }
    if(data_name != train_data.name && data_name != test_data.name) {
        cout << "[GBDT-PL]: unknown data name " << data_name << endl;
    }
    return scores_per_iteration[data_name];
}

void Booster::Predict(DataMat &predict_data, vector<double> &results, int iters) {

    double start_t = omp_get_wtime();
 
    int num_data = predict_data.num_data;  
  
    auto data = predict_data.unbined_data;
    
    if(iters < 0) {
    	iters = booster_config.num_trees;
    }

    results.clear();
    results.resize(booster_config.num_classes * num_data, 0.0);
    if(booster_config.loss == "l2") {
#pragma omp parallel for schedule(static) num_threads(24) 
	for(int i = 0; i < num_data; ++i) {
	    results[i] = avg_label;
	}	
    }

#pragma omp parallel for schedule(static) num_threads(48)
    for(int k = 0; k < predict_data.num_data; ++k) {
    	for(int i = 0; i < iters; ++i) { 
            for(int j = 0; j < booster_config.num_classes; ++j) {
                int offset = j * num_data;
	        results[k + offset] += trees[i * booster_config.num_classes + j]->PredictSingle(data[k]);
            }
    	} 
    }
    double end_t = omp_get_wtime();
    cout << "[GBDT-PL] predict time: " << (end_t - start_t) << endl;
}

void Booster::BoostFromAverage() {
    double average = 0.0;
#pragma omp parallel for schedule(static) num_threads(booster_config.num_threads) reduction(+:average)
    for(int i = 0; i < train_data.num_data; ++i) {
        average += train_data.label[i];
    }
    average /= train_data.num_data;
    avg_label = average;
#pragma omp parallel for schedule(static) num_threads(booster_config.num_threads)
    for(int i = 0; i < train_data.num_data; ++i) {
        train_predict_values[i] = average;
    }
#pragma omp parallel for schedule(static) num_threads(booster_config.num_threads)
    for(int i = 0; i < test_data.num_data; ++i) {
        test_predict_values[i] = average;
    }
}
