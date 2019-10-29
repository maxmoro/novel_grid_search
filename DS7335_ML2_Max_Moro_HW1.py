import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score # other metrics?
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import pdb
import itertools
from scipy import mean
import matplotlib.pyplot as plt
import random

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

#%% Functions

def cartesian(params):
    #creting a new dictionary with the cartesian product of the selected parameters
    #based on https://docs.python.org/3/library/itertools.html
    keys = params.keys()
    vals = params.values()
    for rowVals in itertools.product(*vals):
        yield dict(zip(keys, rowVals))


def gridSearch(clfs,data):
    #main functions that run the grid search across multiple classifiers and multiple parameters
    #results are saved in a dictuare with totals by model, parameters set, 
    #       and details about the CV outputs
	allResults=[]
	for clf, params in clfs.items():
		#pdb.set_trace()
		results = runGrid(a_clf=clf,data=data,params=params) #run the grid with the selected classifier
		bestAcc = max([r['meanAccuracy'] for r in results]) #find the best accurancy within the differnt parameters
		bestAccId = ([r['paramsId'] for r in results if r['meanAccuracy']==bestAcc])[0] #find the ID of the best accuracy
        #save everything in a dictionary
		allResults.append({'model':clf.__name__
                           ,'results':results  #results is a dictionary with the output per each parameter set
                           ,'bestMeanAccuracy':bestAcc
                           ,'bestMeanAccurancId':bestAccId
                           })
	return allResults
    
def runGrid(a_clf, data,params={}):
	random.seed(1701)
	X, Y, n_folds = data # unpack data containter
	kf = KFold(n_splits=n_folds) # Establish the cross validation
	cv = {} # results are saved in a dictionaty
	paramGrid = list(cartesian(params)) #create the list with the cartesian product of all the passed paramters
	paramRes = [] #lsit that will contain the resouls per each parameter set.
	print("-> running model:",a_clf.__name__)
	scaler = preprocessing.StandardScaler()
	for paramId, param in enumerate(paramGrid): #run each parameter set
		print("----> param:",param,end='' )
		cv = {} # results are saved in a dictionaty
		for ids, (train_index, test_index) in enumerate(kf.split(X, Y)): #run by CV
			clf = a_clf(**param) # unpack paramters into clf is they exist
			Xtrain = scaler.fit_transform(X[train_index])
			Xtest = scaler.fit_transform(X[test_index])
			clf.fit(Xtrain, Y[train_index])
			pred = clf.predict(Xtest)
			#create a dictionary with the ouput of each CV
			cv[ids]= {'clf': clf  
                      ,'accuracy': accuracy_score(Y[test_index], pred)
                      ,'param':param
                      ,'train_index': train_index
                      ,'test_index': test_index
                      }
        #save the output of all CV in a dictionary
        #save also the mean accuracy of all the CV
		#pdb.set_trace()
		acc=  mean([r['accuracy'] for r in cv.values()])
		paramRes.append({'paramsId':paramId
                         ,'meanAccuracy':  acc
                         ,'params':param
                         ,'CV':cv
                         })
		print(" = Accuracy:",acc)
	return paramRes
#%% Loading Data

#the file contains data from the list of people that received promotion or not during the past 7 years.
#also contain infomration like tenure, time since last promotion, awards, ect..

filename = 'data_for_classification.csv'
data = np.genfromtxt(filename, delimiter=',',skip_header=1,missing_values=0)[0:5000,:]
Y = data[:,0] #get the Y (first col)
X = np.delete(data,0,1) #get the X all other cols
X = np.nan_to_num(X) #not needed as the data is clean..


#%% Run Classifiers

#the classifieres are stored in a dictiornaty including their specific apramters and selected values
clfs = {RandomForestClassifier: 
            {'n_estimators':{1,4,8}
            ,'max_depth':{10,11,12}
            ,'min_samples_split':{5,10,15}
            }
        ,svm.SVC:
            {'C':{1,5,10}
			,'kernel':{'linear','rbf','poly'}
            ,'tol':{1e-3,1e-1,1e-3}
			,'gamma':{'auto'}
        }
        ,KNeighborsClassifier:
            {'n_neighbors' :{3,5,9}
            ,'algorithm': {'ball_tree','kd_tree','brute'}
            ,'p' : {1,2}
            }
}

n_folds = 5 #5 folds CV

data = (X, Y, n_folds) #packing the data in a tuple

results = gridSearch(clfs, data) #run the codes

print('completed, output in the dictionary: results')
#the resutls are store in result dictionary.
#this dictionary contain 1 row per each model, 
#each model's row contain a dictionary with the best value,it's ID,
#and the outputs per each parameter-set saved as dictionary
#each parameter-set contains also a dictionary per each CV

#%% Organize Output List

plotX=[]
plotXmean=[]
plotY=[]
for m in results:
	for r in m['results']:
		plotX.append((list(cv['accuracy'] for cv in r['CV'].values())))
		plotXmean.append(r['meanAccuracy'])
		plotY.append(m['model'] + ':\n ' + ','.join("{!s}={!r}".format(key,val) for (key,val) in r['params'].items()))
		
sortIdx = list(np.argsort(plotXmean))

plotXmean = [plotXmean[i] for i in sortIdx]       
plotX = [plotX[i] for i in sortIdx]       
plotY = [plotY[i] for i in sortIdx]       

print("Best Classifier")
print(" ",plotY[-1])
print(" Mean Accuracy:",plotXmean[-1])

#%% PLOT
#set figure dimension (based on the number of eleemnts we have)
fig, ax = plt.subplots(figsize=(15,len(plotXmean)*.4))

#setting size of labels
ax.yaxis.label.set_size(20)

#axis labels
ax.set_ylabel("Parameters")
ax.set_xlabel("Accuracy")
ax.tick_params(labelsize=8)
#
fig.subplots_adjust(left=0.4,right=0.9,top=0.95,bottom=0.1)
#box plot
ax.boxplot(plotX,vert=False,labels=plotY,showmeans=True) 

#adding text for the mean point
for id,Xmean in enumerate(plotXmean):
	ax.annotate(round(Xmean,3),xy=(Xmean*1.01,id+0.87),xycoords='data',fontsize='small',color='green')

#saving to file
plt.savefig(fname = 'plot.png')#,bbox_inches='tight')

plt.show()
	
#%%

#flipping the order of the list to print the ranking (I miss ggplot!)
sortIdx = list(reversed(np.argsort(plotXmean)))
plotXmean = [plotXmean[i] for i in sortIdx]       
plotX = [plotX[i] for i in sortIdx]       
plotY = [plotY[i] for i in sortIdx]   

# PARAMETERS
printTop = 999 #<<-- Change this to limit the output list
printToFile = "N" #<<-- Use "N" to write to console, or "Y" to output to ranking_output.txt file
#
if(printToFile != "Y"):
    f = sys.stdout
else:
    filename='outputRanking.txt'
    if os.path.exists(filename):   os.remove(filename)
    f=open(filename, "a")

print("Results from other classifiers and hyperparameters, from the lowest to the highest Mean Accurancy:",file=f)
#loop to write the classifiers ranked from top to bottom
for id, r in enumerate(plotY):
    print("_______________________________",file=f)
    print("Ranking:", id+1,file=f)
    print("Classifier and Parameters",file=f)
    print("",r,file=f)
    print("    Mean Accuracy:",round(plotXmean[id],3),file=f)
    if (id+1) >= printTop: break 

if(printToFile == "Y"): f.close()