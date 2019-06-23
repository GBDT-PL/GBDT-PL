//
//  auc.cpp
//  LinearGBMVector
//


//

#include <stdio.h>
#include "metric.hpp"
#include <parallel/algorithm>

ACC::ACC(const BoosterConfig &booster_config, vector<double> &labelss, vector<double> &scoress):
Metric(booster_config, labelss, scoress) {
    name = "acc"; 
}

double ACC::Eval() {
    int n = static_cast<int>(scores.size());    
    double acc = 0.0;
    if(booster_config.num_classes > 1) {
#pragma omp parallel for schedule(static) num_threads(booster_config.num_threads) reduction(+:acc)
        for(int i = 0; i < n; ++i) {
            double max_score = scores[i];
            int max_class = 0;
            for(int j = 1; j < booster_config.num_classes; ++j) {
                if(scores[j * n + i] > max_score) {
                    max_score = scores[j * n + i];
                    max_class = j;
                }
            }    
            if(max_class == labels[i]) {
                ++acc;
            }
        }   
    }
    else {
#pragma omp parallel for schedule(static) num_threads(booster_config.num_threads) reduction(+:acc)
        for(int i = 0; i < n; ++i) {
            int pred = static_cast<int>(scores[i] > 0);
            if(pred == labels[i]) {
                ++acc;
            }
        }       
    }
    acc /= n;
    return acc;
}
