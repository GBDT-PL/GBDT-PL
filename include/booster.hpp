//
//  booster.hpp
//  LinearGBM
//


//

#ifndef booster_hpp
#define booster_hpp

#include "tree.hpp"
#include <vector>
#include "booster_config.hpp"
#include "datamat.hpp"
#include <string>
#include "metric.hpp"   
#include <string>
#include "data_partition.hpp"
#include "objective.hpp"
#include "alignment_allocator.hpp"

using std::vector;

class Booster {
private:
    std::vector<Tree*> trees;   
    
    BoosterConfig &booster_config;
    
    vector<double> train_predict_values;
    vector<double> test_predict_values;
    
    vector<int> data_indices_tmp;
    
    double avg_label;
    
    struct GradientIndex {
        double gradient;
        int index;
        GradientIndex(double gradientt, int indexx) {
            gradient = gradientt;
            index = indexx;
        }
    };
    
    vector<double> tmp_gradients; 
    
    int num_data_aligned; 
    fvec64 gradients;
    fvec64 sampled_gradients; 
    
    Objective *objective;
    
    Metric *train_eval;
    Metric *test_eval;

    DataMat& train_data;
    
    DataMat& test_data;
    
    double evaluate_time; 
    
    void EvaluateWithPredictValues(std::string metric,
                                   const std::vector<double> &predict_values,
                                   const std::vector<double> &labels,
                                   const std::vector<double> &weights, 
                                   int num_classes,
                                   DataMat* data_mat);
    
    void GetMask(double prob, std::vector<bool>& mask);
    
    double EvalLoss(const vector<double> &predicts,
                    const vector<double> &labels,
                    const vector<double> &weights,
                    DataMat* data_mat); 
    
    int num_threads;
    
    DataPartition data_partition;
    
    vector<double> train_scores, test_scores;
    vector<double> train_probs, test_probs;
    vector<double> train_max_scores, test_max_scores;
    
    void SetupTrees();
    void SetupEvals();
    void SetupObjs();
    void BoostFromAverage();
    
    void Sample(int iteration, int cur_class); 
public:
    
    Booster(BoosterConfig &_booster_config, DataMat &_data_mat, DataMat &_test_data);
    
    void Train();
    
    void Predict(DataMat &predict_data, vector<double> &results); 
};


#endif /* booster_hpp */
