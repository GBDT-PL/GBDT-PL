//
//  regression.cpp
//  LinearGBMVector
//


//

#include "objective.hpp"

Regression::Regression(fvec64 &gradientss, const vector<double> &labelss, const vector<double> &scoress,            
                       const BoosterConfig &booster_config, int num_data):  
Objective(gradientss, labelss, scoress, booster_config, num_data) { 
    
}

void Regression::UpdateGradients(int cur_class) {
#pragma omp parallel for schedule(static) num_threads(booster_config.num_threads)
    for(int i = 0; i < num_data; ++i) {
        gradients[i << 1] = (scores[i] - labels[i]);                                    
        gradients[(i << 1) + 1] = 1.0; 
    }
}

double Regression::EvalLoss() {
    double loss = 0.0;
#pragma omp parallel for schedule(static) num_threads(booster_config.num_threads) reduction(+:loss)
    for(int i = 0; i < num_data; ++i) {
        double diff = scores[i] - labels[i];    
        loss += diff * diff;
    }
    return loss;    
}
