//
//  multiclass.cpp
//  LinearGBMVector
//


//

#include <cmath>
#include "objective.hpp"

MultiClass::MultiClass(fvec64 &gradientss, const vector<double> &labelss, const vector<double> &scoress,    
               const BoosterConfig &booster_config, int num_data, int num_aligned_dataa):
Objective(gradientss, labelss, scoress, booster_config, num_data) {
    num_aligned_data = num_aligned_dataa;
}

void MultiClass::UpdateGradients(int cur_class) {
    if(cur_class == 0) {
#pragma omp parallel for schedule(static) num_threads(booster_config.num_threads)
        for(int i = 0; i < num_data; ++i) {
            double max_score = scores[i];
            for(int j = 1; j < booster_config.num_classes; ++j) {
                if(scores[j * num_data + i] > max_score) {
                    max_score = scores[j * num_data + i];   
                }
            }
            double sum_prob = 0.0;
            for(int j = 0; j < booster_config.num_classes; ++j) {
                sum_prob += exp(scores[j * num_data + i] - max_score);      
            }
            for(int j = 0; j < booster_config.num_classes; ++j) {
                double prob = exp(scores[j * num_data + i] - max_score) / sum_prob;
                gradients[(i << 1) + j * num_aligned_data * 2] = (prob - (j == (int)labels[i]));
                gradients[(i << 1) + 1 + j * num_aligned_data * 2] = 2 * prob * (1 - prob);
            }
        }
    }
}

double MultiClass::EvalLoss() {
    double loss = 0.0;
#pragma omp parallel for schedule(static) num_threads(booster_config.num_threads) reduction(-:loss)
    for(int i = 0; i < num_data; ++i) {
        double sum_of_prob = 0.0;
        double max_score = scores[i];
        for(int j = 1; j < booster_config.num_classes; ++j) {
            if(scores[i + j * num_data] > max_score) {
                max_score = scores[i + j * num_data];
            }
        }
        for(int j = 0; j < booster_config.num_classes; ++j) {
            sum_of_prob += std::exp(scores[i + j * num_data] - max_score);
        }
        double prob = std::exp(scores[i + static_cast<int>(labels[i]) * num_data] - max_score);
        loss -= std::log(prob); 
    }
    return loss;
}
