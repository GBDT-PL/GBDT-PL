*******************************
Steps to Use GBDT-PL Python API
*******************************
To use the python API of GBDT-PL, we need 5 steps:
1. Create a dictionary of parameters for GBDT-PL. 
2. Create training, evaluation and test datasets. All datasets in GBDT-PL are represented by class DataMat. The parameter dictionary is passed to all the DataMat instances.
3. Create a booster.
4. Train the booster.
5. Predict. 
The following sections will introduce each of the above steps. 

*************************
GBDT-PL Python Parameters
*************************
Booster Parameters
==================
* ``num_trees``

  - Number of boosting iterations.
  
* ``num_leaves``

  - Maximum number of leaves in each tree. Used when ``grow_by=leaf``
 
* ``min_sum_hessians``

  - The minimum value of sum of hessians of all the data in a leaf. 

* ``lambda``

  - The coefficient for L2 regularization for the parameters of linear models. 
  
* ``objective``

  - The objective for gradient boosting.
  - Choices: ``logistic``, ``multi-logistic``, ``l2``
  - ``logistic``: For binary classfication.
  - ``multi-logistic``: For multi-class classification.
  - ``l2``: For regression
  
* ``learning_rate``

  - The weight of a single tree in the final ensemble. 
  
* ``eval_metric``

  - The metric used on evaluation sets. 
  - Choices: ``auc``, ``rmse``, ``ndcg``
  - ``auc``: For binary classification.
  - ``rmse``: For regression.
  - ``ndcg``: For ranking.
  
* ``num_classes``

  - The number of classes, used for multi-class classification only. For binary classification and regression, it should be set as ``1``.
  
* ``num_threads``
  - The number of threads used in the parallelism.
  
* ``verbose``
  - Controls the evaluation. 
  - Choices: ``0``, ``1``, ``2``
  - ``0``: No dataset is evaluated.
  - ``1``: Only the evaluation set is evaluated in each boosting step.
  - ``2``: Both the training and evaluation sets are evaluated in each boosting step.
  
* ``boosting_type``
  - Controls the sampling in boosting.
  - Choices: ``gbdt``, ``goss``
  - ``gbdt``: No sampling.
  - ``goss``: Use goss sampling as LightGBM.
  
* ``goss_alpha``
  - The sample ratio by magnitudes of hessians. Used when ``boosting_type=goss``
  
* ``goss_beta``
  - The random sample ratio. Used when ``boosting_type=goss``
Tree Parameters
===============
* ``num_bins``
  - The maximum number of bins used in the histograms. Currently, we only support at most 255 bins.
  
* ``min_gain``
  - The minimum gain reduction to allow a split.
  
* ``max_var``
  - The maximum number of features used in the linear models.
  
* ``grow_by``
  - The approach to grow a tree. 
  - Choices: ``leaf``, ``level``
  - ``leaf``: Grow the tree in a leaf-wise manner.
  - ``level``: Grow the tree in a level-wise manner.
  
* ``leaf_type``
  - The approach to fit parameters in leaves.
  - Choices: ``constant``, ``linear``, ``additive_linear``
  - ``constant``: Use ordinary regression trees with constant leaf values.
  - ``linear``: Use linear functions to produce the leaf values. The parameters of linear functions are all recalculated when a leaf is split into child nodes.
  - ``additive_linear``: Same as ``linear``, but we don't recalculate all the linear function parameters. We use the half-additive fitting technique in our paper to refit the linear functions.
  
* ``max_depth``
  - The maximum number of levels in each tree, used when ``grow_by=level``

***************
Create Datasets
***************
Each dataset in GBDT-PL is represented by a DataMat instance. To create a DataMat, we use the following function.

.. code:: sh

    
