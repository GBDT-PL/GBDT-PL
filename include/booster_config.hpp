//
//  booster_config.hpp
//  LinearGBM
//


//

#ifndef booster_config_hpp
#define booster_config_hpp

#include <string>
#include <iostream>

using std::cout;
using std::endl;
using std::string;

class BoosterConfig {
public:
    BoosterConfig(int _num_trees,
                  int _max_leaf,
                  double _min_gain,
                  double _l2_reg,
                  std::string _loss,
                  double _learning_rate,
                  std::string _eval_metric,
                  int _num_classes,
                  int _num_threads,
                  int _max_bin,
                  double _min_sum_hessian_in_leaf,
                  int _num_vars,
                  std::string _grow_by,
                  std::string _leaf_type,               
                  int _max_level,
                  int _verbose,
                  double _sparse_threshold,
                  string _boosting_type,
                  double _goss_alpha,
                  double _goss_beta):
    num_trees(_num_trees),
    max_leaf(_max_leaf),
    min_gain(_min_gain),
    loss(_loss),
    learning_rate(_learning_rate),
    eval_metric(_eval_metric),
    num_classes(_num_classes),
    num_threads(_num_threads),
    max_bin(_max_bin),
    min_sum_hessian_in_leaf(_min_sum_hessian_in_leaf),
    num_vars(_num_vars),
    grow_by(_grow_by),
    max_level(_max_level),
    l2_reg(_l2_reg),
    verbose(_verbose),
    leaf_type(_leaf_type),
    sparse_threshold(_sparse_threshold),
    boosting_type(_boosting_type),
    goss_alpha(_goss_alpha),
    goss_beta(_goss_beta)
     {
         
        all_prepare_bin_time = 0.0;
        all_prepare_histogram_time = 0.0;
        all_find_split_time = 0.0;
        all_split_time = 0.0;
         all_update_train_score_time = 0.0;
         all_update_gradients_time = 0.0;
         all_after_train_tree_time = 0.0;
         
         prepare_bin_time = 0.0;
         prepare_histogram_time = 0.0;
         find_split_time = 0.0;
         split_time = 0.0;
         update_train_score_time = 0.0;
         update_gradients_time = 0.0;
         after_train_tree_time = 0.0;       
         
        cout << "num_trees " << num_trees << endl;
        cout << "max_leaf " << max_leaf << endl;
        cout << "min_gain " << min_gain << endl;
        cout << "l2_reg " << l2_reg << endl;
        cout << "loss " << loss << endl;
        cout << "learning_rate " << learning_rate << endl;  
        cout << "eval_metric " << eval_metric << endl;
        cout << "num_classes " << num_classes << endl;
        cout << "num_threads " << num_threads << endl;
        cout << "max_bin " << _max_bin << endl;
        cout << "min_sum_hessian_in_leaf " << min_sum_hessian_in_leaf << endl;
        cout << "num_vars " << num_vars << endl;
        cout << "grow_by " << grow_by << endl;
        cout << "max_level " << max_level << endl;
        cout << "verbose " << verbose << endl;
         cout << "leaf type " << _leaf_type << endl;
         cout << "sparse threshold 1 " << sparse_threshold << endl;
         cout << "boosting_type " << boosting_type << endl;
         cout << "goss_alpha " << goss_alpha << endl;
         cout << "goss_beta " << goss_beta << endl;
    }
    
    int num_trees;
    
    int max_leaf;
    
    double min_gain;
    
    double l2_reg;
    
    std::string loss;
    
    double learning_rate;
    
    std::string eval_metric;
    
    int num_classes;
    
    bool train_multi_class;
    
    int num_threads;
    
    int max_bin;
    
    double min_sum_hessian_in_leaf;
    
    int num_vars;
    
    int max_level;
    
    int verbose;
    
    std::string grow_by;
    
    std::string leaf_type;
    
    double sparse_threshold;
    
    double all_prepare_bin_time;
    double prepare_bin_time;
    double all_prepare_histogram_time;  
    double prepare_histogram_time;
    double all_find_split_time;
    double find_split_time;
    double all_update_gradients_time;
    double prepare_local_data_time;
    double all_split_time;
    double split_time;
    double update_train_score_time;
    double all_update_train_score_time;
    double update_gradients_time;
    double all_after_train_tree_time;
    double after_train_tree_time;
    
    string boosting_type;
    double goss_alpha;
    double goss_beta; 
    
    void ClearIterationTime() {
        prepare_bin_time = 0.0;
        prepare_histogram_time = 0.0;
        find_split_time = 0.0;
        prepare_local_data_time = 0.0;
        split_time = 0.0;
        update_train_score_time = 0.0;
        update_gradients_time = 0.0;
        after_train_tree_time = 0.0;
    }
    
    void PrintAllTime() {
        cout << "all_prepare_bin_time " << all_prepare_bin_time << endl;
        cout << "all_prepare_histogram_time " << all_prepare_histogram_time << endl;
        cout << "all_find_split_time " << all_find_split_time << endl;
        cout << "all_split_time " << all_split_time << endl;
        cout << "all_update_gradients_time " << all_update_gradients_time << endl;
        cout << "all_update_train_score_time " << all_update_train_score_time << endl;
        cout << "all_after_train_tree_time " << all_after_train_tree_time << endl;      
    }
    
    void PrintIterationTime() {
        cout << "prepare_bin_time " << prepare_bin_time << endl;
        cout << "prepare_histogram_time " << prepare_histogram_time << endl;
        cout << "find_split_time " << find_split_time << endl;
        cout << "split_time " << split_time << endl;
        cout << "update_gradients_time " << update_gradients_time << endl;
        cout << "update_train_score_time " << update_train_score_time << endl;
        cout << "after_train_tree_time " << after_train_tree_time << endl;
    }
};


#endif /* booster_config_hpp */
