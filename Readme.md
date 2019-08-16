# Applied Machine Learning with Python

## About this course
This course will introduce the learner to applied machine learning, focusing more on the techniques and methods than on the statistics behind these methods. The course will start with a discussion of how machine learning is different than descriptive statistics, and introduce the scikit learn toolkit through a tutorial. The issue of dimensionality of data will be discussed, and the task of clustering data, as well as evaluating those clusters, will be tackled. Supervised approaches for creating predictive models will be described, and learners will be able to apply the scikit learn predictive modelling methods while understanding process issues related to data generalizability (e.g. cross validation, overfitting). The course will end with a look at more advanced techniques, such as building ensembles, and practical limitations of predictive models. By the end of this course, students will be able to identify the difference between a supervised (classification) and unsupervised (clustering) technique, identify which technique they need to apply for a particular dataset and need, engineer features to meet that need, and write python code to carry out an analysis.

## What is covered in this course
* Understand basic ML concepts and workflow
* How to properly apply 'black-box' machine learning
components and features
* Learn how to apply machine learning algorithms in
Python using the scikit-learn package
## What is not covered
* Underlying theory of statistical machine learning
* Lower-level details of how particular ML components work
* In-depth material on more advanced concepts like deep learning
## Introduction:
### What is Machine Learning (ML)?
* The study of computer programs (algorithms)
that can learn by example
* ML algorithms can generalize from existing
examples of a task

### Types of Machine Learning:

1. #### Supervised ML
   Learn to predict target values from labelled data.
    1. ##### Classification: Target values with discrete classes.
        ###### eg: Differentiate between fruit types
    2. ##### Regression: Target values are continuous values.
        ###### eg: Predict house price
        
2. #### Unsupervised ML
    Find structure in unlabeled data
    1. ##### Clustering: Find groups of similar instances in the data.
        ###### eg: Finding clusters of similar users 
    2. ##### Outlier detection: Detecting abnormal server access patterns.
        ###### eg: Predict house price

### Basic Machine Learning Workflow
1. #### Representation
    Choose:
    - A feature representation
    - Type of classifier to use
        ###### e.g. image pixels, with k-nearest neighbor classifier
2. #### Evaluation
    Choose:
    * What criterion distinguishes good vs. bad classifiers?
        ###### e.g. % correct predictions on test set
3. #### Optimization
    Choose:
    * How to search for the settings/parameters that give the best classifier for this evaluation criterion.
        ###### e.g. try a range of values for "k" parameter in k-nearest neighbor classifier

### Setting up environment:
1. #### scikit-learn: Python Machine Learning Library
    1. ##### scikit-learn Homepage: http://scikit-learn.org/
    2. ##### scikit-learn User Guide: http://scikit-learn.org/stable/user_guide.html
    3. ##### scikit-learn API reference: http://scikit-learn.org/stable/modules/classes.html
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    ```
1. #### Python Environment with below libraries:
    1. ##### SciPy Library: Scientific Computing Tools
        * Provides a variety of useful scientific computing     tools, including statistical distributions,     optimization of functions, linear algebra, and a    variety of specialized mathematical functions.
        * With scikit-learn, it provides support for sparse     matrices, a way to store large tables that consist  mostly of zeros.
        ```python
        import scipy as sp
        ```
    2. ##### NumPy: Scientific Computing Library
        * Provides fundamental data structures used by  scikit-learn, particularly multi-dimensional arrays.
        * Typically, data that is input to scikit-learn will    be in the form of a NumPy array. 
        hat consist  mostly of zeros.
        ```python
        import numpy as np
        ```
    3. ##### Pandas: Data Manipulation and Analysis
        * Provides key data structures like DataFrame
        * Also, support for reading/writing data in different   formats
        hat consist  mostly of zeros.
        ```python
        import pandas as pd
        ```
    4. #### matplotlib and other plotting libraries
        * We typically use matplotlib's pyplot module:
        ```python
        import matplotlib.pyplot as plt
        ```
        * We also sometimes use the seaborn visualization
        library (http://seaborn.pydata.org/)
        ```python
        import seaborn as sn
        ```
        * And sometimes the graphviz plotting library:
        ```python
        import graphviz
        ```
    5. #### Libraries versions used in this course
        | Library name  | Minimum version |
        | -             | -
        | scikit-learn  | 0.18.1
        | scipy         | 0.19.0
        | numpy         | 1.12.1
        | pandas        | 0.19.2
        | matplotlib    | 2.0.1
        | seaborn       | 0.7.1
        | graphviz      | 0.7

