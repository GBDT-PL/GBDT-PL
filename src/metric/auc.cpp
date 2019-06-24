//
//  auc.cpp
//  LinearGBMVector
//


//

#include <stdio.h>
#include "metric.hpp"
#include <parallel/algorithm>

AUC::AUC(const BoosterConfig &booster_config, vector<double> &labelss, vector<double> &scoress):
Metric(booster_config, labelss, scoress) {
    name = "auc"; 
    larger_is_better = true;
}

static bool compare_rel(ScoreIndexRel *a, ScoreIndexRel *b) {
    return (a->score < b->score);
}

double AUC::Eval() {
    int n = static_cast<int>(scores.size());
    std::vector<ScoreIndexRel*> pairs(n, nullptr);
#pragma omp parallel for schedule(static) num_threads(booster_config.num_threads)
    for(int i = 0; i < n; ++i) {
        pairs[i] = (new ScoreIndexRel(scores[i], i, labels[i]));
    }
    __gnu_parallel::sort(pairs.begin(), pairs.end(), compare_rel);
    
    std::vector<double> x_dots(n + 1, 0.0), y_dots(n + 1, 0.0);
    
    int true_instances = 0, false_instances = 0;
#pragma omp parallel for schedule(static) num_threads(booster_config.num_threads) \
    reduction(+:true_instances,false_instances)
    for(int i = 0; i < n; ++i) {
        if(labels[i] == 1.0) {
            true_instances += 1;
        }
        else {
            false_instances += 1;
        }
    }
    
    x_dots[0] = 0.0;
    y_dots[0] = 1.0;
    
    double auc = 0.0;
    
    vector<double> thread_true_positive(booster_config.num_threads + 1, 0.0);
    vector<double> thread_true_negative(booster_config.num_threads + 1, 0.0);
    
    thread_true_positive[0] = true_instances;
    
#pragma omp parallel for schedule(static) num_threads(booster_config.num_threads)
    for(int i = 0; i < n; ++i) {
        int tid = omp_get_thread_num();
        if(pairs[i]->rel == 1.0) {
            thread_true_positive[tid + 1] -= 1;
        }
        else {
            thread_true_negative[tid + 1] += 1;
        }
    }
    
    for(int i = 1; i < booster_config.num_threads + 1; ++i) {  
        thread_true_positive[i] += thread_true_positive[i - 1];
        thread_true_negative[i] += thread_true_negative[i - 1];
    }
    
#pragma omp parallel for schedule(static) num_threads(booster_config.num_threads)
    for(int i = 0; i < n; ++i) {
        int tid = omp_get_thread_num();
        if(pairs[i]->rel == 1.0) {
            thread_true_positive[tid] -= 1;
        }
        else {
            thread_true_negative[tid] += 1;
        }
        x_dots[i] = (thread_true_negative[tid] * 1.0 / false_instances);
        y_dots[i] = (thread_true_positive[tid] * 1.0 / true_instances);
    }
    
#pragma omp parallel for schedule(static) num_threads(booster_config.num_threads) reduction(+:auc)
    for(int i = 0; i < n; ++i) {
        auc += (y_dots[i] + y_dots[i + 1]) * (x_dots[i + 1] - x_dots[i]) / 2;   
    }
    
#pragma omp parallel for schedule(static) num_threads(booster_config.num_threads)
    for(int i = 0; i < n; ++i) {
        delete pairs[i];
    }
    
    return auc;
}
