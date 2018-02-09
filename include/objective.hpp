//
//  objective.hpp
//  LinearGBMVector
//


//

#ifndef objective_hpp
#define objective_hpp

#include <stdio.h>
#include <iostream>
#include <vector>
#include "alignment_allocator.hpp"  
#include <omp.h>
#include "booster_config.hpp"

using std::vector;

//objective is used to update gradients and hessians, and prepare values for histogram construction
class Objective {
protected:
    //reference to gradients in data_partition
    fvec64 &gradients;
    const vector<double> &labels;   
    const vector<double> &scores;
    int num_data;
    const BoosterConfig &booster_config;    
public:
    Objective(fvec64 &gradientss, const vector<double> &labelss, const vector<double> &scoress,
              const BoosterConfig &booster_config,
              int num_dataa);
    
    virtual void UpdateGradients(int cur_class) = 0;
    virtual double EvalLoss() = 0;
};

class Binary : public Objective {
private:
public:
    Binary(fvec64 &gradientss, const vector<double> &labelss, const vector<double> &scoress,                
           const BoosterConfig &booster_config, int num_dataa);
    
    void UpdateGradients(int cur_class);
    double EvalLoss();
};

class Regression : public Objective {
private:
public:
    Regression(fvec64 &gradientss, const vector<double> &labelss, const vector<double> &scoress,
               const BoosterConfig &booster_config, int num_dataa);
    
    void UpdateGradients(int cur_class);
    double EvalLoss();
};

class MultiClass : public Objective {
private:
    int num_aligned_data; 
public:
    MultiClass(fvec64 &gradientss, const vector<double> &labelss, const vector<double> &scoress,
               const BoosterConfig &booster_config, int num_dataa, int num_aligned_dataa); 
    
    void UpdateGradients(int cur_class);
    double EvalLoss(); 
};

#endif /* objective_hpp */
