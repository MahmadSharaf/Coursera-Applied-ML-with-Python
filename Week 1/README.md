# Week 1 Assignments

## Quiz Questions

1. **Training a model using labeled data and using this model to predict the labels for new data is known as**
    * Supervised Learning
2. **Modeling the features of an unlabeled dataset to find hidden structure is known as**
    * Unsupervised Learning
3. **Training a model using categorically labelled data to predict labels for new data is known as**
    * Classification
4. **Training a model using labelled data where the labels are continuous quantities to predict labels for new data is known as**
    * Regression
5. **Using the data for classes 0, 1, and 2 plotted below, what class would a KNeighborsClassifier classify the new point as for k = 1 and k = 3?**
    * k=1: Class 1 k=3: Class 2
6. **Which of the following is true for the nearest neighbor classifier (Select all that apply):**  
   [`WrongAnswers`]
    * ~~Partitions observations into k clusters where each observation belongs to the cluster with the nearest mean~~
    * ~~A higher value of k leads to a more complex decision boundary~~
7. **Why is it important to examine your dataset as a first step in applying machine learning?**
    * See what type of cleaning or preprocessing still needs to be done
    * You might notice missing data
    * Gain insight on what machine learning model might be appropriate, if any
    * Get a sense of how difficult the program might be
8. **The key purpose of splitting the dataset into training and test sets is:**
    * To estimate how well the learned model will generalize to new data
9. **The purpose of setting the random_state parameter in train_test_split is:**
    * To make experiments easily reproducible by always using the same partitioning of the data
10. **Given a dataset with 10,000 observations and 50 features plus one label, what would be the dimensions of X_train, y_train, X_test, and y_test? Assume a train/test split of 75%/25%.**
    * X_train: (7500, 50)  
    y_train: (7500, )  
    X_test: (2500, 50)  
    y_test: (2500, )

## Programming assignment

### Q1: Convert the sklearn.dataset *cancer* to a DataFrame

This function should return a (569, 31) DataFrame

### A1

1. The function that converts sklearn.dataset to pandas     DataFrame is

    ```python
    pd.DataFrame(data,columns)
    ```

2. Cancer Data misses the target info, so they needed to be appended together

    ```python
    data = np.column_stack((cancer.data,cancer.target))
    ```

3. Cancer Columns also misses the target.

    ```python
    columns = np.append(cancer['feature_names'],'target')
    ```

4. The function now would be like:

    ```python
    def answer_one():

    columns = np.append(cancer['feature_names'],'target')
    data = np.column_stack((cancer.data,cancer.target))

    return pd.DataFrame(data=data, columns=columns)
    ```

    **OR in one line code**

    ```python
    def answer_one():
    return pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                  columns= np.append(cancer['feature_names'], ['target']))
    ```

### Q2: What is the class distribution? (i.e. how many instances of malignant (encoded 0) and how many benign (encoded 1)?)

This function should return a Series named target of length 2 with integer values and index = ['malignant', 'benign']

### A2

1. There are two ways to count:
    * Direct:

    ```Python
    count_0 = cancerdf.groupby('target').count().iloc[0,0]
    count_1 = cancerdf.groupby('target').count().iloc[1,0]
    ```

    * Indirect:

    ```Python
    count_0 = len(cancerdf[cancerdf['target']==0])
    count_1 = len(cancerdf[cancerdf['target']==1])
    ```

2. Then append them to a pandas Series

    ```Python
    target = pd.Series((count_0,count_1),index=['malignant', 'benign'])
    ```

### Q3: Split the DataFrame into X (the data) and y (the labels)

This function should return a tuple of length 2: (X, y), where

* *X, a pandas DataFrame, has shape (569, 30)*
* *y, a pandas Series, has shape (569,).*

### A3

```python
def answer_three():
    cancerdf = answer_one()
    X,y = cancerdf.iloc[:,:-1],cancerdf['target']

    return X,y
```

### Q4: Using train_test_split, split X and y into training and test sets (X_train, X_test, y_train, and y_test)

This function should return a tuple of length 4: (X_train, X_test, y_train, y_test), where

* X_train has shape (426, 30)
* X_test has shape (143, 30)
* y_train has shape (426,)
* y_test has shape (143,)

### A4

Function `train_test_split(data, label, random_state)` takes the data and the labels and split them into 75% for trainings and 25% for tests, in which random_state takes an integer value that guarantees the function will return the same randomization.

```python
from sklearn.model_selection import train_test_split

def answer_four():
    X, y = answer_three()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    return X_train, X_test, y_train, y_test
```

### Q5: Using KNeighborsClassifier, fit a k-nearest neighbors (knn) classifier with X_train, y_train and using one nearest neighbor (n_neighbors = 1)

This function should return a **sklearn.neighbors.classification.KNeighborsClassifier**

### A5

1. KNeighborsClassifier(n_neighbors):  
    * It is a type of machine learning algorithm.
    * It can be used for classification and regression.
    * It is an example of what's called instance based or memory based supervised learning.
    * It returns a classifier that classifies the input with respect to the nearest n_neighbors neighbors that is the most predominant.
2. knn.fit() trains the model to memorize all its features and labels

    ```python
    from sklearn.neighbors import KNeighborsClassifier

    def answer_five():
        X_train, X_test, y_train, y_test = answer_four()
        knn = KNeighborsClassifier(n_neighbors = 1)

        return knn.fit(X_train, y_train)
    ```

### Q6: Using your knn classifier, predict the class label using the mean value for each feature

### A6

You can use ***cancerdf.mean()[:-1].values.reshape(1, -1)*** which gets the mean value for each feature, ignores the target column, and reshapes the data from 1 dimension to 2 (necessary for the predict method of KNeighborsClassifier).

*This function should return a numpy array either array([ 0.]) or array([ 1.]).*

```python
def answer_six():
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    knn=answer_five()

    return knn.predict(means)
```

### Q7: Using your knn classifier, predict the class labels for the test set X_test

This function should return a numpy array with shape (143,) and values either 0.0 or 1.0.

### A7

```python
def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()

    return knn.predict(X_test)
```

### Q8: Find the score (mean accuracy) of your knn classifier using X_test and y_test

This function should return a float between 0 and 1

### A8

```python
def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    score = knn.score(X_test,y_test)

    return score
```
