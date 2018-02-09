//
//  rmse.cpp
//  LinearGBMVector
//


//

#include <stdio.h>
#include "metric.hpp"

RMSE::RMSE(const BoosterConfig &booster_config, vector<double> &labelss, vector<double> &scoress):
Metric(booster_config, labelss, scoress) {
    name = "rmse"; 
}

double RMSE::Eval() {
    double loss = 0.0;
    int num_data = static_cast<int>(labels.size());
#pragma omp parallel for schedule(static) num_threads(booster_config.num_threads) reduction(+:loss)
    for(int i = 0; i < labels.size(); ++i) {
        double diff = scores[i] - labels[i];
        loss += diff * diff;
    }
    return std::sqrt(loss / num_data);  
}
