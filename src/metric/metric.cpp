//
//  metrics.cpp
//  LinearGBM
//


//

#include "metric.hpp"   
Metric::Metric(const BoosterConfig &booster_configg, vector<double> &labelss, vector<double> &scoress):
booster_config(booster_configg), labels(labelss), scores(scoress) {} 
