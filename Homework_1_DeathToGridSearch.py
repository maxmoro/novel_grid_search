import numpy as np
from sklearn.metrics import accuracy_score # other metrics?
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
# adapt this code below to run your analysis

# Recommend to be done before live class 2
# 1. Write a function to take a list or dictionary of clfs and hypers ie use logistic regression, each with 3 different sets of hyper parrameters for each

# Recommend to be done before live class 3
# 2. expand to include larger number of classifiers and hyperparmater settings
# 3. find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings

# Recommend to be done before live class 4
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function

M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])
L = np.ones(M.shape[0])
n_folds = 5

data = (M, L, n_folds)

def run(a_clf, data, clf_hyper={}):
  M, L, n_folds = data # unpack data containter
  kf = KFold(n_splits=n_folds) # Establish the cross validation
  ret = {} # classic explicaiton of results

  for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
    clf = a_clf(**clf_hyper) # unpack paramters into clf is they exist

    clf.fit(M[train_index], L[train_index])

    pred = clf.predict(M[test_index])

    ret[ids]= {'clf': clf,
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)}
  return ret

results = run(RandomForestClassifier, data, clf_hyper={})
