import math

import matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt
import  sklearn.linear_model as sk
import pandas as pd
import sklearn.metrics as me
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import seaborn as sea
from sklearn import metrics




#"""Xtest = np.load('Xtest_Regression2.npy')
#Ytrain = np.load('Ytrain_Regression2.npy')
#Xtrain = np.load('Xtrain_Regression2.npy')
#Accuracy = {}"""

#"We will apply feature selection with 3 different techniques and will analyze the results."

#"Select k best"

#"""selector = SelectKBest(f_classif, k=8)
#selector.fit(Xtrain, Ytrain.ravel())
#Xtrain_skb = Xtrain[:, selector.get_support()]
#print(selector.get_support())
#print(Xtrain_skb.shape)"""

#"Recursive feature elimination:"

#selector = RFECV(SGDClassifier(random_state=1), step=1)
#selector.fit(Xtrain, Ytrain.ravel().astype('int'))
#Xtrain_rfe4 = Xtrain[:, selector.get_support()]
#selector = RFECV(SGDClassifier(random_state=1), step=1, min_features_to_select=8)
#selector.fit(Xtrain, Ytrain.ravel().astype('int'))
#Xtrain_rfe8 = Xtrain[:, selector.get_support()]
#print(selector.get_support())
#print(Xtrain_rfe8.shape)
#print(Xtrain_rfe4.shape)

#"Select from model:"

#selector = SelectFromModel(SGDClassifier(random_state=1), threshold='mean')
#selector.fit(Xtrain, Ytrain.ravel().astype('int'))
#Xtrain_sfm = Xtrain[:, selector.get_support()]
#print(selector.get_support())
#print(Xtrain_sfm.shape)

#"Huber regressor: The huber regressor is a regressor that is very tolerant to outliers."

#def huber(Xtrain):
#  huber = HuberRegressor().fit(Xtrain, Ytrain.ravel())
#  scoreCVH = cross_val_score(huber, Xtrain, Ytrain.ravel()).mean()
#  print(scoreCVH)
#huber(Xtrain)
#huber(Xtrain_rfe4)
#huber(Xtrain_rfe8)
#huber(Xtrain_skb)
#huber(Xtrain_sfm)"""

#"We see that the recursive feature elimination technique gives the " \
# best results, when only 4 variables are chosen."\
#"But if we only take 4 features out of the 10, we are doing overfitting. In the other cases, " \
#"the results are worse or the same as without feature selection, " \
#"so we will not apply feature selection in this case."



#"""huber = HuberRegressor().fit(Xtrain, Ytrain.ravel())
#scoreCVH = cross_val_score(huber, Xtrain, Ytrain.ravel()).mean()
#Accuracy["Huber"] =  scoreCVH
#print(scoreCVH)

#"Robustscaler: The robustscaler normalisation is a normalisation technique that is very tolerant to outliers."

#"X_robust = RobustScaler().fit_transform(Xtrain)"

#   model = LinearRegression()
#model.fit(X_robust, Ytrain)
#scoreCVLS = cross_val_score(model, X_robust, Ytrain).mean()
#print(scoreCVLS)
#Accuracy["LS"] =  scoreCVLS"""

#"Robustscaler: The robustscaler normalisation is a normalisation technique that is very tolerant to outliers."

#"X_robust = RobustScaler().fit_transform(Xtrain)"


#"LS"

#   model = LinearRegression()
#model.fit(X_robust, Ytrain)
#scoreCVLS = cross_val_score(model, X_robust, Ytrain).mean()
#print(scoreCVLS)
#Accuracy["LS"] =  scoreCVLS


#"Ridge:"

#points = np.logspace(-6, 6, 13)
# print(points)
#for i in range(3):
#    modelRCV = RidgeCV(alphas=points)
#    modelRCV.fit(X_robust, Ytrain)
    # print(modelRCV.alpha_)
#    index = list(points).index(modelRCV.alpha_)
#    lowerbound = list(points)[index - 1]
#    upperbound = list(points)[index + 1]
 #   points = np.linspace(lowerbound, upperbound, 100)
    # print(points)"""

#scoreCVR = cross_val_score(modelRCV, X_robust, Ytrain).mean()
#print(scoreCVR)
#Accuracy["Ridge"] = scoreCVR

#"Lasso:"

#points = np.logspace(-6, 6, 13)
# print(points)
#for i in range(3):
#    modelLCV = LassoCV(alphas=points)
#    modelLCV.fit(X_robust, Ytrain.ravel())
#     print(modelRCV.alpha_)
#    index = list(points).index(modelLCV.alpha_)
#    lowerbound = list(points)[index - 1]
#    upperbound = list(points)[index + 1]
#    points = np.linspace(lowerbound, upperbound, 100)
    # print(points)"""

#scoreCVL = cross_val_score(modelLCV, X_robust, Ytrain.ravel()).mean()
#print(scoreCVLS)
#Accuracy["Lasso"] = scoreCVL"""

#"print(list(Accuracy.keys())[list(Accuracy.values()).index(max(Accuracy.values()))])"

#"The results are really bad, so we try to implement a technique ourselves:"


#"First we calculated the residuals|Y-Ypred| and searched which values were out of the interquartile range" \
#"But this method catched a low number of outliers(only the most extreme ones) and gave poor results"


#"The method that we ended up using to resolve this problem, was a method created by us, were we recursevely "
#"apply a regression line to our data set and take out the biggest residual"
#"To decide when to stop we took in our consideration several facts: "
#"     1. The data set has about 20% of outliers "
#"     2. The max of the scores(SSE) given bY Cross Validation "
#"     3. The standart deviaton of the scores (SSE) given by cross validation"
#"     4. The possibility of our model taking out  inliers"
#"We used 2. and 3. to study the perfomance and reaction of the model to the removal of points we were taking points "
#"Because of 4 we suppose an error of 10% on the detection of outliers so we can be more confident that we take all the outliers"
#"We used 1. and 4. to decide an upper limit of search in this case taking out of our data (percentage outlier + error on detection) lines maximum"



Xtest_final = np.load("Xtest_regression2.npy")
Y = (np.load('Ytrain_Regression2.npy'))
X = np.load('Xtrain_Regression2.npy')





num_out = 0
residuals = []
out = []
index = []
scores = []
cvmean = []
cvstd = []
perc_outlier = Y.size//5
error_method = perc_outlier//10



while num_out <= perc_outlier + error_method :
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    residuals = []
    lin = sk.LinearRegression()
    Fit = lin.fit(X,Y)
    predict = lin.predict((X))
    y_pred = Fit.predict(X_test)
    cross = cross_val_score(Fit, X_train, y_train, scoring="neg_mean_squared_error", cv=45)
    score = lin.score(X, Y)
    scores.append(score)
    cvmean.append(np.max(-cross*np.size(cross)))
    cvstd.append(np.std(-cross * np.size(cross)))

    for k in range(Y.size):
        residuals.append(math.dist(Y[k], predict[k]))
    max = 0
    max_index = -1
    for i in range(Y.size):
        if residuals[i] > max:
            max = residuals[i]
            max_index = i
    Y = np.delete(Y, max_index, axis=0)
    X = np.delete(X, max_index, axis=0)
    num_out +=1
print(scores)
print(cvmean)
print(cvstd)
print(Y.size)
predition = lin.predict(Xtest_final)





