//
//  objective.cpp
//  LinearGBMVector
//


//

#include "objective.hpp"

Objective::Objective(fvec64 &gradientss, const vector<double> &labelss, const vector<double> &scoress,  
                     const BoosterConfig &booster_configg, int num_dataa):
gradients(gradientss), labels(labelss), scores(scoress), num_data(num_dataa), booster_config(booster_configg) {}
