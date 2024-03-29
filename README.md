# GBDT-PL
We extend gradient boosting to use piecewise linear regression trees (PL Trees), 
instead of piecewise constant regression trees. PL Trees can accelerate convergence of
GBDT. Moreover, our new algorithm fits better to modern computer architectures with powerful
Single Instruction Multiple Data (SIMD) parallelism. We name our new algorithm GBDT-PL.

## Experiments 
We evaluate our algorithm on 10 public datasets.


|Dataset Name| #Training | #Testing | #Features |      Task      | Link |
|------------|------------|----------|-----------|----------------|------|
|    Higgs   | 10,000,000 | 500,000  |     28    | Classification | [higgs](https://archive.ics.uci.edu/ml/datasets/HIGGS) |
|   Epsilon  | 400,000 | 100,000  |     2000    | Classification | [epsilon](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) |
|  HEPMASS   | 7,000,000 | 3,500,000 | 28 | Classification | [hepmass](https://archive.ics.uci.edu/ml/datasets/HEPMASS)|
|  SUSY   | 4,000,000 | 1,000,000 | 18 | Classification | [susy](https://archive.ics.uci.edu/ml/datasets/SUSY)|
| CASP | 29,999 | 15,731 | 9 | Regression | [casp](https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure) |
| SGEMM | 193,280 | 48,320 | 14 | Regression | [sgemm](https://archive.ics.uci.edu/ml/datasets/SGEMM+GPU+kernel+performance) |
| SUPERCONDUCTOR | 17,008 | 4,255 | 81 | Regression | [superconductor](https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data) |
| CT | 29,999 | 15,731 | 384 | Regression | [ct](https://archive.ics.uci.edu/ml/datasets/Relative+location+of+CT+slices+on+axial+axis) |
| Energy | 29,999 | 15,788 | 27 | Regression | [energy](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency) |
| Year | 412,206 | 103,139 | 90 | Regression | [year](https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD) |

### Accuracy 
We compare the accuracy in the following table. For XGBoost, LightGBM and CatBoost different settings of hyperparameters are tried and the best result is picked for each algorithm. For GBDT-PL, we separte 20% of the training data for validation. Details of settings can be found in **Parameter Setting** Section of [Experiment Setting](https://github.com/GBDT-PL/GBDT-PL/blob/master/docs/supple.pdf).

|Dataset Name| LightGBM | XGBoost | CatBoost |      GBDT-PL      | 
|------------|------------|----------|-----------|----------------|
|    Higgs   | 0.854025 | 0.854147  |     0.851590    | **0.860198** | 
|   Epsilon  | 0.951422 | 0.948292  |     0.957327    | **0.957894** | 
|  HEPMASS   | 0.95563 | 0.95567 | 0.95554 | **0.95652** | 
|  SUSY   | 0.878112 | 0.877825 | 0.878206 | **0.878287** |
| CASP | 3.4961 | 3.4939 | 3.5183 | **3.4574** | 
| SGEMM | 4.61431 | 4.37929 | 4.41177 | **4.16871** | 
| SUPERCONDUCTOR | 8.80776 | 8.91063 | **8.78452** | 8.79527 | 
| CT | 1.30902 | 1.34131 | 1.36937 | **1.23753** | 
| Energy | **64.256** | 64.780 | 65.761 | 65.462 | 
| Year | 8.38817 | 8.37935 | 8.42593 | **8.37233** | 

### Training Time
We use the histogram version of XGBoost and LightGBM. Each tree in our experiments has at most **255 leaves**. For CatBoost, we constrain the **maximum tree depth to 8**. The **learning rate is 0.1**. We test both **63 bins and 255 bins** for the histograms. **lambda** is the coefficient for regularization terms, we set it as 0.01. **min sum of hessians** is the minimum sum of hessians allowed in each leaf. We set it as 100 to prevent the trees from growing too deep. For GBDT-PL, we use at most **5 regressors** per leaf. 

|max leaves | learning rate | bins | lambda | min sum of hessians |
|-----------|---------------|------|-----------|---------------------|
|255 | 0.1 | 63/255 | 0.01 | 100 |

We plot the accuracy of testing sets w.r.t. training time in Figure 2. To leave out the effect of evaluation time, for each dataset we have two separate runs. In the first run we record the training time per iteration only, without doing evaluation. In the second round we evaluate the accuracy every iteration. We use 24 threads for all algorithms. The preprocessing time is excluded. 

![](https://github.com/GBDT-PL/GBDT-PL/raw/master/figures/training-time-1.png)
![](https://github.com/GBDT-PL/GBDT-PL/raw/master/figures/training-time-2.png)
![](https://github.com/GBDT-PL/GBDT-PL/raw/master/figures/training-time-3.png)

### Convergence Rate
We run 500 iterations and plot testing accuracy per iteration.  For classification tasks, we use AUC as metric, and for regression we use RMSE.
![](https://github.com/GBDT-PL/GBDT-PL/raw/master/figures/convergence-1.png) 
![](https://github.com/GBDT-PL/GBDT-PL/raw/master/figures/convergence-2.png) 
![](https://github.com/GBDT-PL/GBDT-PL/raw/master/figures/convergence-3.png) 
