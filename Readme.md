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
    * Feature normalization:
      * MinMax Scaling: transform all the input variables, so they're all on the same scale between zero and one.

        ```python
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(x_train)   # compute the min and max feature   values for each feature in this training dataset.
        X_trained_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        clf = Ridge().fit(X_train_scaled, y_train)
        r2_score = clf.score(X_test_scaled, y_test)
        # or more efficiently fit and transform in one step
        X_train_scaled = scaler.fit_transform(X_train)
        ```

        Critical aspects to feature normalization:
        1. Apply the same scalar object to both training and testing data.
        2. Training the scalar object on the training data and not on the test data. If it trained on the test data, it will cause a phenomena called Data Leakage, where the training phase has information that is leaked from the test set.
    * Polynomial Features
    ![Polynomial-Features_equation](/images/Polynomial&#32;features&#32;equation.jpg)
      * Generate new features consisting of all polynomial combinations of the original two features 𝑥0,𝑥1.
      * The degree of the polynomial specifies how many variables participate at a time in each new feature (above example: degree 2).
      * This is still a weighted linear combination of features, so it's still a linear model, and can use same least-squares estimation method for w and b.
      * Adding these extra polynomial features allows us a much richer set of complex functions that we can use to fit to the data.
      * This intuitively as allowing polynomials to be fit to the training data instead of simply a straight line, but using the same least-squares criterion that minimizes mean squared error.
      * We want to transform the data this way to capture interactions between the original features by adding them as features to the linear model.
      * Polynomial feature expansion with high as this can lead to complex models that over-fit.
      * Polynomial feature expansion is often combined with a regularized learning method like ridge regression.

        ```python
        from sklearn.linear_model import LinearRegression
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import PolynomialFeatures

        X_train, X_test, y_train, y_test =  train_test_split(X_F1    y_F1, random_state = 0)

        linreg = LinearRegression().fit(X_train, y_train)

        poly = PolynomialFeatures(degree=2)
        X_F1_poly = poly.fit_transform(X_F1)

        X_train, X_test, y_train, y_test =  train_test_split(X_F1_poly,y_F1, random_state = 0)

        linreg = LinearRegression().fit(X_train, y_train)
        ```

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
        ![equation](images/Linear&#32;equation.jpg)  
        <!-- $$
        \hat{y} = \hat{w_0}x_0 + \hat{w_1} x_1 + ... \hat{w_n} x_n + \hat{b}
        $$ -->
        The hat(^) is an indication that the parameter is estimated during training process.  
        **y**: the predicted output.  
        **w_i**: the model coefficients or feature weights.  
        **b**: the biased term or intercept of the model.

        w, b parameters are estimated by:

        * They are estimated from training the data.
        * There are different methods correspond to different 'fit' criteria and goals and ways to control complexity.
        * `Squared loss function` returns the squared difference between the target value and the  actual value as the penalty.
        * The learning algorithm then computes or searches for the set of w, b parameters that optimize an objective function, typically to minimize the total of this loss function over all training points.
            1. Least Squares:
            ![Least-squares_equation](images/Least-squares&#32;Equation.jpg)
               * The most popular way to estimate w and b parameters is using what's called least-squares linear regression or ordinary least-squares. Least-squares finds the values of w and b that minimize the total sum of squared differences between the predicted y value and the actual y value in the training set. Or equivalently it minimizes the mean squared error of the model.
               * This technique is designed to find the slope, the w value, and the b value of the y intercept, that minimize this squared error, this mean squared error.
               * The mean squared error is the square difference between predicted and actual values, and then all these are added up, and then divided by the number of training points, take the average, that will be the mean squared error of the model.
               * One thing to note about this linear regression model is that there are no parameters to control the model complexity. No matter what the value of w and b, the result is always going to be a straight line. This is both a strength and a weakness of the model. 

                    ```python
                    from sklearn.linear_model import LinearRegression

                    X_train, X_test, y_train, y_test= train_test_split(X_R1, y_R1, random_state= 0)

                    linreg = LinearRegression().fit(X_train, y_train)

                    # w_i: coefficients
                    linreg.coef_
                    # b: the intercept term
                    linreg.intercept_

                    #! In Scikit-Learn object attribute ends with an underscore, this means that this attribute is derived from the training data, not quantities that set by the user.
                    ```

            2. Ridge Regression:  
               ![ridge_equation](/images/Ridge&#32;Equation.jpg)
               * Ridge regression uses the same least-squares criterion, but with one difference. During the training phase, it adds a penalty for large feature weights in w parameters.
               * Once the parameters are learned, its prediction formula is the same as ordinary least-squares.
               * The addition of a parameter penalty is called regularization. Regularization prevents over fitting by restricting the model, typically to reduce its complexity.
               * It uses L2 regularization: minimize sum of squares of w entries.
               * If ridge regression finds two possible linear models that predict the training data values equally well, it will prefer the linear model that has a smaller overall sum of squared feature weights.
               * The amount of regularization to apply is controlled by the alpha parameter. Larger alpha means more regularization and simpler linear models with weights closer to zero.(default 1.0)

                    ```python
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()

                    from sklearn.linear_model import Ridge
                    X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state=0)

                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    linridge = Ridge(alpha = 20.0).fit(X_train_scaled, y_train)
                    ```

            3. Lasso Regression
                ![Lasso_equation](/images/Lasso&#32;Equation.jpg)
                * Like ridge regression, lasso regression adds a regularization penalty term to the ordinary least-squares objective, that causes the model W-coefficients to shrink towards zero.
                * Lasso regression is another form of regularized linear regression that uses an L1 regularization penalty for training (instead of ridge's L2 penalty).
                * L1 Penalty: minimizes the sum of the absolute values of the coefficients.
                * This has the effect of setting parameter weights in w to zero for the least influential variables. This called a sparse solution: a kind of feature selection.
                * The parameter alpha controls the amount of L1 regularization (default = 1.0).
                * The prediction formula is the same as ordinary least-squares.

                    ```python
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()

                    from sklearn.linear_model import Ridge
                    X_train, X_test, y_train, y_test =  train_test_split(X_crime, y_crime, random_state=0)

                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    linlasso = Lasso(alpha = 2.0, max_iter = 10000) .fit(X-train_scaled, y_train)
                    ```

                * When to use ridge vs lasso:
                  * Many small/medium sized effects: use ridge.
                  * Only a few variables with medium/large effect: use lasso.

    2. Logistic Regression  
        ![Flowchart box](images/Logistic&#32;Regression&#32;flow&#32;chart.jpg)  
        ![Logistic fn](images/Logistic&#32;Regression&#32;function.jpg)
       * It is a kind of generalized linear model.
       * In spite of being called a regression measure, it is actually used for classification
       * like ordinary least squares and other regression methods, logistic regression takes a set input variables, the features, and estimates a target value.
       * Unlike ordinary linear regression, in it's most basic form logistic repressions target value is a binary variable instead of a continuous value.
       * There are flavors of logistic regression that can also be used in cases where the target value to be predicted is a multi class categorical variable, not just binary.
       * Logistic regression is similar to linear regression, but with one critical addition. The logistic regression model still computes a weighted sum of the input features xi and the intercept term b (like in linear regression), but it runs this result through a special non-linear function f, the logistic function represented by this new box in the middle of the diagram to produce the output y. The effect of applying the logistic function is to compress the output of the linear function so that it's limited to a range between 0 and 1. Below the diagram, you can see the formula for the predicted output y hat which first computes the same linear combination of the inputs xi, model coefficient weights wi hat and intercept b hat, but runs it through the additional step of applying the logistic function to produce y hat.
       * If we pick different values for b hat and the w hat coefficients, we'll get different variants of this s shaped logistic function, which again is always between 0 and 1.
       * To perform logistic, regression in Scikit-Learn, you import the logistic regression class from the sklearn.linear model module, then create the object and call the fit method using the training data just as you did for other class files like k nearest neighbors.

            ```python
            from sklearn.linear_model import LogisticRegression

            X_train, X_test, y_train, y_test = train_test_split(X_C2,   y_C2,random_state = 0)
            clf = LogisticRegression(C=1).fit(X_train, y_train)
            ```

         * L2 regularization is 'on' by default (like ridge regression)
         * Parameter C controls amount of regularization (default 1.0)
         * As with regularized linear regression, it can be important to normalize all features so that they are on the same scale.

## Aspects to be considered

1. ### Overfitting

    Informally, overfitting typically occurs when we try to fit a complex model with an inadequate amount of training data. And overfitting model uses its ability to capture complex patterns by being great at predicting lots and lots of specific data samples or areas of local variation in the training set. But it often misses seeing global patterns in the training set that would help it generalize well on the unseen test set.

2. ### Underfitting

    The model is too simple for the actual trends that are present in the data. It doesn't even do well on the training data and thus, is not at all likely to generalize well to test data.

* To avoid these, the below points would help:
  1. First, try to draw the data with respect to the labels and try to figure out the relationship between, whether its linear, quadratic, polynomial and so on.
