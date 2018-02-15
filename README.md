# GBDT-PL
We extend gradient boosting to use piecewise linear regression trees (PL Trees), 
instead of piecewise constant regression trees. PL Trees can accelerate convergence of
GBDT. Moreover, our new algorithm fits better to modern computer architectures with powerful
Single Instruction Multiple Data (SIMD) parallelism. We name our new algorithm GBDT-PL.

## Experiments 
We evaluate our algorithm on 4 public datasets. In addition, we create 2 synthetic datasets. 

|Dataset Name| #Training | #Testing | #Features |      Task      | Link |
|------------|------------|----------|-----------|----------------|------|
|    Higgs   | 10,000,000 | 500,000  |     28    | Classification | [higgs](https://archive.ics.uci.edu/ml/datasets/HIGGS) |
|   Epsilon  | 400,000 | 100,000  |     2000    | Classification | [epsilon](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) |
|  HEPMASS   | 7,000,000 | 3,500,000 | 28 | Classification | [hepmass](https://archive.ics.uci.edu/ml/datasets/HEPMASS)|
| CASP | 29,999 | 15,731 | 9 | Regression | [casp](https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure) |
| Poly | 2,000,000 | 1,000,000 | 200 | Regression | TBA |
| Cubic | 10,000,000 | 1,000,000 | 10 | Regression | TBA |

For Higgs, we use the first 10,000,000 data points as training set, and the reset as testing set. For Epsilon, we use the first 400,000 as training and the reset as testing. For HEPMASS, we use the first 7,000,000 as training and the reset as testing. For CASP, we use the first 29,999 as training and the reset as testing. 

### Convergence Rate
We run 500 iterations and plot testing accuracy per iteration. Figure 1 shows the results. We use lgb for [LightGBM](https://github.com/Microsoft/LightGBM) and xgb for [XGBoost](https://github.com/dmlc/xgboost) in the figure legends. 
Following table shows the testing accuracy after 500 iterations. 

|Algorithm | HIGGS | HEPMASS | CASP | Epsilon | Cubic | Poly |
|----------|-------|---------|------|---------|-------|------|
|GBDT-PL, 63 bins | 0.8539 | 0.9563 | 3.6497 | 0.9537 | 0.5073 | 879.7 |
|LightGBM, 63 bins | 0.8455 | 0.9550 | 3.6217 | 0.9498 | 0.9001 | 923.2 |
|XGBoost, 63 bins | 0.8449 | 0.9549 | 3.6252 | 0.9498 | 0.9010 | 921.4 |
|GBDT-PL, 255 bins | 0.8545 | 0.9563 | 3.6160 | 0.9537 | 0.2686 | 673.4 |
|LightGBM, 255 bins | 0.8453 | 0.9550 | 3.6206 | 0.9499 | 0.5743 | 701.3 |
|XGBoost, 255 bins | 0.8456 | 0.9550 | 3.6177 | 0.9500  | 0.5716 | 701.1 |

![](https://github.com/GBDT-PL/GBDT-PL/raw/master/figures/convergence.png) 

### Training Time
We plot the accuracy of testing sets w.r.t. training time in Figure 2. To leave out the effect of evaluation time, for each dataset we have two separate runs. In the first run we record the training time per iteration only, without doing evaluation. In the second round we evaluate the accuracy every iteration. We use 24 threads for all algorithms. The preprocessing time is excluded. 

![](https://github.com/GBDT-PL/GBDT-PL/raw/master/figures/training-time.png)

### Testing Time
To measure the testing time accurately, we expand some testing sets by replicating the original ones several times. Note that replicating the testing sets won't change the testing accuracy (at least for AUC and RMSE). The details are listed in the following table. We plot the testing accuracy w.r.t. the testing time in Figure 3. We use 48 threads for all experiments in this section.

|Dataset Name | #Original Testing | #Expanded Testing |  Replicate   | 
|-------------|-------------------|-------------------|--------------|
|    Higgs   | 500,000 | 10,000,000  |     x20    |
|   Epsilon  | 400,000 | 400,000  |     x1    | 
|  HEPMASS   | 3,500,000 | 3,500,000 | x1 | 
| CASP | 15,731 | 8,652,050 | x550 |
| Poly | 1,000,000 | 1,000,000 | x1 | 
| Cubic | 1,000,000 | 10,000,000 | x10 | 

![](https://github.com/GBDT-PL/GBDT-PL/raw/master/figures/testing-time.png)
