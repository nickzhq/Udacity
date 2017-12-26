#!/usr/bin/python

import sys
import pickle
import math
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'bonus', 'long_term_incentive', 'deferred_income', 
'deferral_payments', 'exercised_stock_options', 'restricted_stock', 
'restricted_stock_deferred', 'total_stock_value', 
'from_this_person_to_poi', 'from_messages','from_poi_to_this_person', 'to_messages'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
del data_dict['LOCKHART EUGENE E']
del data_dict['THE TRAVEL AGENCY IN THE PARK']
del data_dict['TOTAL']
'''
### Task 3: Create new feature(s)
for key in data_dict.keys():
	data_dict[key]['rate_from_poi'] = float( data_dict[key]['from_this_person_to_poi'] ) / float( data_dict[key]['from_messages'] )
	data_dict[key]['rate_to_poi']   = float( data_dict[key]['from_poi_to_this_person'] ) / float( data_dict[key]['to_messages'] )
	# print data_dict[key]['rate_from_poi'], "  ------  ", data_dict[key]['rate_to_poi'] 
	if math.isnan( data_dict[key]['rate_from_poi'] ):
		data_dict[key]['rate_from_poi'] = 0
	if math.isnan( data_dict[key]['rate_to_poi'] ):
		data_dict[key]['rate_to_poi']  = 0
	# print data_dict[key]['rate_from_poi'], "  ------  ", data_dict[key]['rate_to_poi']
	del data_dict[key]['from_this_person_to_poi']
	del data_dict[key]['from_messages']
	del data_dict[key]['from_poi_to_this_person']
	del data_dict[key]['to_messages']
'''
# drop the useless features
features_list.remove('from_this_person_to_poi')
features_list.remove('from_messages')
features_list.remove('from_poi_to_this_person')
features_list.remove('to_messages')
# add the new features
#features_list.append('rate_from_poi')
#features_list.append('rate_to_poi')
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# parameter score including new features
from sklearn.feature_selection import SelectKBest
fs = SelectKBest(k="all").fit(features, labels)
print "features scores: ", fs.scores_
'''
features scores:  [ 17.43208738  19.98987553   9.47251714  11.04775908   0.24851659
  23.96833213   8.81209393   0.06709599  23.32934371  15.71478942
   2.88258498]
Based on the scores, k should be 6
'''
poi = 0
for i in labels:
	if i == True:
		poi += 1
print 'poi: ', poi
print 'non-poi: ', len(labels) - poi

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
'''
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.4, random_state=42)
'''
# cross validation
from sklearn.model_selection import StratifiedShuffleSplit
features_train = []
features_test  = []
labels_train   = []
labels_test    = []
cv = StratifiedShuffleSplit(n_splits=3, test_size=0.4, random_state = 42)
for train_idx, test_idx in cv.split(features, labels): 
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )

# first try 
'''
from sklearn.feature_selection import RFE  
from sklearn.linear_model import LinearRegression  

names = sorted(['salary', 'deferred_income', 'deferral_payments',
'bonus', 'total_stock_value', 'expenses', 'exercised_stock_options', 'restricted_stock_deferred',
'long_term_incentive', 'restricted_stock', 'rate_from_poi', 'rate_to_poi']) 
#use linear regression as the model  
clf = LinearRegression()  
#rank all features, i.e continue the elimination until the last one  
rfe = RFE(clf, n_features_to_select=1)  
rfe.fit(features_train, labels_train)  
  
print ("Features sorted by their rank:")  
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))  
'''
# second try
'''
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
from sklearn.model_selection import GridSearchCV
param = [{'splitter': ('best', 'random'),
          'min_samples_split': [2,3,4,5,6,7,8,9,10],
          'max_depth': [None, 10, 30, 50, 70, 100] }]
tr = GridSearchCV( clf, param, cv=10 )
tr.fit( features_train, labels_train )
print "best: ", tr.best_score_
print "best_param_: ", tr.best_params_

clf = DecisionTreeClassifier(min_samples_split=tr.best_params_['min_samples_split'], 
	                         splitter=tr.best_params_['splitter'], 
	                         max_depth=tr.best_params_['max_depth'])
'''
# third try
'''
from sklearn.svm import SVC
clf = SVC()
from sklearn.model_selection import GridSearchCV
param = [{'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
          'C': [0.1, 1, 10, 100, 1000, 10000],
          'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000] }]
tr = GridSearchCV( clf, param)
tr.fit( features_train, labels_train )
print "best: ", tr.best_score_
print "best_param_: ", tr.best_params_

clf = SVC(kernel=tr.best_params_['kernel'], C=tr.best_params_['C'], gamma=tr.best_params_['gamma'])
'''
# optimizing parameters
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([('sc', StandardScaler()),
	             ('pca', PCA()),
	             ('clf', GaussianNB()) ])
param_grid = dict(pca__n_components = [1,2,3,4,5,6,7,8,9] )
'''
param_grid = dict(pca__n_components = [1,2,3,4,5,6,7,8,9], 
	              clf__splitter = ['best', 'random'],
	              clf__max_depth = [None, 10, 30, 50, 70, 100],
	              clf__min_samples_split = [2,3,4,5,6,7,8,9,10])
'''
grid_search = GridSearchCV(pipe, param_grid, cv = 10)
grid_search.fit( features_train, labels_train )

print "best: ", grid_search.best_score_
print "best_params_: ", grid_search.best_params_

test_classifier(pipe, my_dataset, features_list )
clf = pipe

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)