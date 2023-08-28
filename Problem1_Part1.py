"""Code from google colabs"""

import numpy as np
import matplotlib.pyplot as plt

"""we will first take a look at the data:"""

Xtest_final = np.load('Xtest_Regression1.npy')
y = np.load('Ytrain_Regression1.npy')
X = np.load('Xtrain_Regression1.npy')

n = X.shape[1]
plt.figure(figsize=(20, 35))
for i in range(n):
    plt.subplot(5, 2, i + 1)
    plt.scatter(X[:, i], y)

"""Now we will divide our X and y into a train set and a test set. We set the random state to 1 to get the same results
every time we run the code"""

from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=1)

"""We will create a dictionary for the prediction accuracy:"""

Accuracy = {}

"""We will now start with a simple linear regression:"""

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(Xtrain, ytrain)

"""print(model.coef_)"""

"""We see that the 6th and the 8th feature are the most important, what is also visible on the graphics. We also see 
that some features are useless. We will remove the less important feature, than the second less important one...and 
see what happen. We use the leave one out cross validation technique to evaluate the performance."""

Xtrain2 = Xtrain[:, 0:9]

Xtrain3 = Xtrain[:, 0:8]

Xtrain4 = np.concatenate((Xtrain[:, 0].reshape(80, 1), Xtrain[:, 2:8]), axis=1)

Xtrain5 = Xtrain[:, 2:8]

Xtrain6 = np.concatenate((Xtrain[:, 2:6], Xtrain[:, 7].reshape(80, 1)), axis=1)

from sklearn.model_selection import cross_val_score

score1 = cross_val_score(model, Xtrain, ytrain).mean()
score2 = cross_val_score(model, Xtrain2, ytrain).mean()
score3 = cross_val_score(model, Xtrain3, ytrain).mean()
score4 = cross_val_score(model, Xtrain4, ytrain).mean()
score5 = cross_val_score(model, Xtrain5, ytrain).mean()
score6 = cross_val_score(model, Xtrain5, ytrain).mean()

"""print(score1, score2, score3, score4, score5)"""

"""We see that the performance increase until Xtrain5 and then decrease with Xtrain6. Nevertheless we will choose 
Xtrain3 because we find that deleting 2 features out of 10 is already enough."""

model = LinearRegression()
model.fit(Xtrain3, ytrain)
scoreLSCV = cross_val_score(model, Xtrain3, ytrain).mean()
Accuracy["LS"] = scoreLSCV

"""Now let's try with ridge. There is already a function to evaluate ridge for all values of alpha and to choose the 
best one: RidgeCV, we test it with some different k's for the k-fold cross-validation. """

from sklearn.linear_model import Ridge, RidgeCV

modelRCV = RidgeCV(alphas=np.logspace(-6, 6, 13), cv=5)
modelRCV.fit(Xtrain3, ytrain)
"""print(modelRCV.alpha_)"""

modelRCV = RidgeCV(alphas=np.logspace(-6, 6, 13), cv=10)
modelRCV.fit(Xtrain3, ytrain)
"""print(modelRCV.alpha_)"""

modelRCV = RidgeCV(alphas=np.logspace(-6, 6, 13))
modelRCV.fit(Xtrain3, ytrain)
"""print(modelRCV.alpha_)"""

"""We see that the best alpha depends on which k we use for the k-fold cross-validation, that's why we will iterate 
over the different k to know which alpha is the most common. """

alphas = []
for i in range(2, 40):
    modelRCV = RidgeCV(alphas=np.logspace(-6, 6, 13), cv=i)
    modelRCV.fit(Xtrain3, ytrain)
    alphas.append(modelRCV.alpha_)
modelRCV = RidgeCV(alphas=np.logspace(-6, 6, 13))
modelRCV.fit(Xtrain3, ytrain)
alphas.append(modelRCV.alpha_)

most_common_alpha = max(set(alphas), key=alphas.count)
"""print(most_common_alpha)"""

plt.plot(alphas)

"""We see that the most common alpha is 0.1. We will now see around 0.1 what ridgeCV returns."""

modelRCV = RidgeCV(alphas=np.linspace(0.01, 1, 100))
modelRCV.fit(Xtrain3, ytrain)
"""print(modelRCV.alpha_)"""

"""So the best alpha is 0.06, we will test the performance with this alpha:"""

modelR = Ridge(alpha=0.06)
modelR.fit(Xtrain3, ytrain)
scoreCVR = cross_val_score(modelR, Xtrain3, ytrain).mean()
Accuracy["Ridge"] = scoreCVR

"""LASSO"""

from sklearn.linear_model import Lasso, LassoCV

"""As we know there are some features that are not useful for the solution. A good idea is to use Lasso, it will make 
automatic feature selection and could give a better solution. Lasso also already has a function that makes cross 
validation automatically. """

alphas = []
for i in range(2, 40):
    modelLCV = LassoCV(alphas=np.logspace(-6, 6, 13), cv=i)
    modelLCV.fit(Xtrain, ytrain.ravel())
    alphas.append(modelLCV.alpha_)
modelLCV = LassoCV(alphas=np.logspace(-6, 6, 13))
modelLCV.fit(Xtrain, ytrain.ravel())
alphas.append(modelLCV.alpha_)
most_common_alpha = max(set(alphas), key=alphas.count)
plt.plot(alphas)
dic = {}
for elem in alphas:
    dic[f"{elem}"] = alphas.count(elem)

"""print(dic)"""

"""We see that the best value for alpha is 0.001, with 17 occurrences, the second one is 0.0001 with 12 occurrences.
These alphas are really low, that gives us the clue that Lasso should not be used in this exercise. We still compute
the model to see its performance:"""

modelL = Lasso(alpha=0.001)
modelL.fit(Xtrain, ytrain)
scoreCVL = cross_val_score(modelL, Xtrain, ytrain).mean()

Accuracy["Lasso"] = scoreCVL

print(Accuracy)

"""We print the best model:"""

print(list(Accuracy.keys())[list(Accuracy.values()).index(max(Accuracy.values()))])

"""We see that the LS and the ridge method have the best scores and have approximately the same score, so we decide 
to use the ridge method to solve the problem. We test it on the test set, without the two last features: """

Xtest3 = Xtest[:, 0:8]
scoreR = modelR.score(Xtest3, ytest)


print(scoreR)

"""That is an accuracy of 0.9982, which is satisfying. We now predict the ytest_final, with the given Xtest_final,
without the last two features:"""

Xtest_final3 = Xtest_final[:, 0:8]
ytest_final = modelR.predict(Xtest_final3)
print(ytest_final.shape)

np.save("ytest.npy", ytest_final)
