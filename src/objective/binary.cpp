//
//  binary.cpp
//  LinearGBMVector
//


//

#include "objective.hpp"
#include <cmath>

Binary::Binary(fvec64 &gradientss, const vector<double> &labelss, const vector<double> &scoress,
               const BoosterConfig &booster_config, int num_data):
Objective(gradientss, labelss, scoress, booster_config, num_data) {
    
}

void Binary::UpdateGradients(int cur_class) {
#pragma omp parallel for schedule(static) num_threads(booster_config.num_threads)
    for(int i = 0; i < num_data; ++i) {
        double prob = 1.0 / (1 + exp(-scores[i]));  
        gradients[i << 1] = (prob - labels[i]);
        gradients[(i << 1) + 1] = prob * (1 - prob);    
    }
}

double Binary::EvalLoss() {
    double loss = 0.0;
#pragma omp parallel for schedule(static) num_threads(booster_config.num_threads) reduction(-:loss)
    for(int i = 0; i < labels.size(); ++i) {
        double prob = 1.0 / (1.0 + std::exp(-scores[i]));
        if(labels[i] == 0.0) {
            loss -= std::log(1 - prob); 
        }
        else {
            loss -= std::log(prob); 
        }
    }
    return loss; 
}
