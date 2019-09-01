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
* In-depth material on more advanced concepts like deep learning.

## Setting up environment

1. ### `scikit-learn`: Python Machine Learning Library

    1. scikit-learn Homepage: <http://scikit-learn.org/>
    2. scikit-learn User Guide: <http://scikit-learn.org/stable/user_guide.html>
    3. scikit-learn API reference: <http://scikit-learn.org/stable/modules/classes.html>

        ```python
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
        ```

2. ### Python Environment with below libraries

    1. `SciPy` Library: Scientific Computing Tools

        * Provides a variety of useful scientific computing     tools, including statistical distributions,     optimization of functions, linear algebra, and a    variety of specialized mathematical functions.
        * With scikit-learn, it provides support for sparse     matrices, a way to store large tables that consist  mostly of zeros.

            ```python
            import scipy as sp
            ```

    2. `NumPy`: Scientific Computing Library

        * Provides fundamental data structures used by  scikit-learn, particularly multi-dimensional arrays.
        * Typically, data that is input to scikit-learn will    be in the form of a NumPy array.
        hat consist  mostly of zeros.

            ```python
            import numpy as np
            ```

    3. `Pandas`: Data Manipulation and Analysis

        * Provides key data structures like DataFrame
        * Also, support for reading/writing data in different   formats
        hat consist  mostly of zeros.

            ```python
            import pandas as pd
            ```

    4. `matplotlib` and other plotting libraries

        * We typically use matplotlib's pyplot module:

            ```python
            import matplotlib.pyplot as plt
            ```

        * We also sometimes use the seaborn visualization
        library (<http://seaborn.pydata.org/)>

            ```python
            import seaborn as sn
            ```

        * And sometimes the graphviz plotting library:

            ```python
            import graphviz
            ```

    5. Libraries versions used in this course

        | Library name | Minimum version |
        | ------------ | --------------- |
        | scikit-learn | 0.18.1          |
        | scipy        | 0.19.0          |
        | numpy        | 1.12.1          |
        | pandas       | 0.19.2          |
        | matplotlib   | 2.0.1           |
        | seaborn      | 0.7.1           |
        | graphviz     | 0.7             |

## Introduction

### Definition Machine Learning (ML)

* The study of computer programs (algorithms)
that can learn by example
* ML algorithms can generalize from existing
examples of a task

### Types of Machine Learning

1. #### Supervised ML

   Learn to predict target values from labelled data.

    1. Classification: Target values with discrete classes  
        eg: Differentiate between fruit types
    2. Regression: Target values are continuous values.  
        eg: Predict house price

2. #### Unsupervised ML

    Find structure in unlabeled data

    1. Clustering: Find groups of similar instances in the data.  
        eg: Finding clusters of similar users.
    2. Outlier detection: Detecting abnormal server access patterns.  
        eg: Predict house price

### Basic Machine Learning Workflow

1. Representation  
    Choose:
    * A feature representation
    * Type of classifier to use  
        e.g. image pixels, with k-nearest neighbor classifier
2. Evaluation  
    Choose:
    * What criterion distinguishes good vs. bad classifiers?  
        e.g. % correct predictions on test set
3. Optimization  
    Choose:
    * How to search for the settings/parameters that give the best classifier for this evaluation criterion.  
        e.g. try a range of values for "k" parameter in k-nearest neighbor classifier

## Getting Started

1. Look at what kind of feature preprocessing is typically needed.

    * Notice missing data.
    * Gain insight on what machine learning model might be appropriate, if any.
    * Get a sense of how difficult the program might be.

2. Split dataset into features X and labels y

3. Split X and y with a percentage for training the algorithm and the remaining for testing its accuracy

    This can be achieved in python as below:

    ```python
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test =
    train_test_split(X, y, random_state=0)
    ```

## Supervised Machine Learning Algorithms

The supervised aspect refers to the need for each training example to have a label in order for the algorithm to learn how to make accurate predictions on future examples.
This is in contrast to unsupervised machine learning where we don't have labels for the training data examples, and we'll cover unsupervised learning in a later part of this course.

1. ### K-nearest neighbor

    * It is a type of machine learning algorithm.
    * It can be used for classification and regression.
    * It is an example of what's called instance based or memory based supervised learning.
    * It returns a classifier that classifies the input with respect to the nearest n_neighbors neighbors that is the most predominant.
     K-nearest neighbors doesn't make a lot of assumptions about the structure of the data and gives potentially accurate, but sometimes unstable predictions
    * It can be sensitive to small changes in the training data.
    * It can be used in python as below:
        1. Initiate a variable as

            ```python
            knn = KNeighborsClassifier(n_neighbors)
            ```

        2. Train the model to memorize all its features and labels

            ```python
            knn.fit()
            ```

        3. To predict a label use the below function with 1 parameter that has the same number of feature as the trained data

            ```python
            knn.predict(param)
            ```

        4. The accuracy can be tested by passing testing data and testing labels

            ```python
            knn.score(X_test, y_test)
            ```

2. ### Linear Model

    * Linear models make strong assumptions about the structure of the data.
    * The target value can be predicted just using a weighted sum of the input variables, a linear function.
    * It can get stable, but potentially inaccurate predictions.

    1. Linear Regression

        $$
        \hat{y} = \hat{w_0}x_0 + \hat{w_1} x_1 + ... \hat{w_n} x_n + \hat{b}
        $$
        $\hat{y}$: the predicted output.  
        $\hat{w_i}$: values are model coefficients or weights.  
        $\hat{b}$: the biased term or intercept of the model.

        $\hat{w}, \hat{b}$ parameters are estimated by:

        * `Squared loss function` returns the squared difference between the target value and the  actual value as the penalty.
        * The learning algorithm then computes or searches for the set of $\hat{w}, \hat{b}$ parameters that minimize the total of this loss function over all training points.
            1. Least Squares:  
               * The most popular way to estimate $\hat{w}$ and $\hat{b}$ parameters is using what's called least-squares linear regression or ordinary least-squares. Least-squares finds the values of $\hat{w}$ and $\hat{b}$ that minimize the total sum of squared differences between the predicted $\hat{y}$ value and the actual $\hat{y}$ value in the training set. Or equivalently it minimizes the mean squared error of the model.
            2. Ridge Regression:
               * Ridge regression uses the same least-squares criterion, but with one difference. During the training phase, it adds a penalty for large feature weights in $\hat{w}$ parameters.
               * Once the parameters are learned, its prediction formula is the same as ordinary least-squares.
               * The addition of a parameter penalty is called regularization. Regularization prevents over fitting by restricting the model, typically to reduce its complexity.

## Aspects to be considered

1. ### Overfitting

    Informally, overfitting typically occurs when we try to fit a complex model with an inadequate amount of training data. And overfitting model uses its ability to capture complex patterns by being great at predicting lots and lots of specific data samples or areas of local variation in the training set. But it often misses seeing global patterns in the training set that would help it generalize well on the unseen test set.

2. ### Underfitting

    The model is too simple for the actual trends that are present in the data. It doesn't even do well on the training data and thus, is not at all likely to generalize well to test data.

* To avoid these, the below points would help:
  1. First, try to draw the data with respect to the labels and try to figure out the relationship between, whether its linear, quadratic, polynomial and so on.
