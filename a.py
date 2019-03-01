# coding: utf-8

# *Python Machine Learning 2nd Edition* by [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017
# 
# Code Repository: https://github.com/rasbt/python-machine-learning-book-2nd-edition
# 
# Code License: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)

# # Python Machine Learning - Code Examples

# # Chapter 3 - A Tour of Machine Learning Classifiers Using Scikit-Learn

# Note that the optional watermark extension is a small IPython notebook plugin that I developed to make the code reproducible. You can just skip the following line(s).

# In[1]:




# In[2]:


from sklearn import __version__ as sklearn_version
from distutils.version import LooseVersion

if LooseVersion(sklearn_version) < LooseVersion('0.18'):
    raise ValueError('Please use scikit-learn 0.18 or newer')


# *The use of `watermark` is optional. You can install this IPython extension via "`pip install watermark`". For more information, please see: https://github.com/rasbt/watermark.*

# ### Overview

# - [Choosing a classification algorithm](#Choosing-a-classification-algorithm)
# - [First steps with scikit-learn](#First-steps-with-scikit-learn)
#     - [Training a perceptron via scikit-learn](#Training-a-perceptron-via-scikit-learn)
# - [Modeling class probabilities via logistic regression](#Modeling-class-probabilities-via-logistic-regression)
#     - [Logistic regression intuition and conditional probabilities](#Logistic-regression-intuition-and-conditional-probabilities)
#     - [Learning the weights of the logistic cost function](#Learning-the-weights-of-the-logistic-cost-function)
#     - [Training a logistic regression model with scikit-learn](#Training-a-logistic-regression-model-with-scikit-learn)
#     - [Tackling overfitting via regularization](#Tackling-overfitting-via-regularization)
# - [Maximum margin classification with support vector machines](#Maximum-margin-classification-with-support-vector-machines)
#     - [Maximum margin intuition](#Maximum-margin-intuition)
#     - [Dealing with the nonlinearly separable case using slack variables](#Dealing-with-the-nonlinearly-separable-case-using-slack-variables)
#     - [Alternative implementations in scikit-learn](#Alternative-implementations-in-scikit-learn)
# - [Solving nonlinear problems using a kernel SVM](#Solving-nonlinear-problems-using-a-kernel-SVM)
#     - [Using the kernel trick to find separating hyperplanes in higher dimensional space](#Using-the-kernel-trick-to-find-separating-hyperplanes-in-higher-dimensional-space)
# - [Decision tree learning](#Decision-tree-learning)
#     - [Maximizing information gain – getting the most bang for the buck](#Maximizing-information-gain-–-getting-the-most-bang-for-the-buck)
#     - [Building a decision tree](#Building-a-decision-tree)
#     - [Combining weak to strong learners via random forests](#Combining-weak-to-strong-learners-via-random-forests)
# - [K-nearest neighbors – a lazy learning algorithm](#K-nearest-neighbors-–-a-lazy-learning-algorithm)
# - [Summary](#Summary)



# In[3]:


from IPython.display import Image


# # Choosing a classification algorithm

# ...

# # First steps with scikit-learn

# Loading the Iris dataset from scikit-learn. Here, the third column represents the petal length, and the fourth column the petal width of the flower samples. 
#The classes are already converted to integer labels where 0=Iris-Setosa, 1=Iris-Versicolor, 2=Iris-Virginica.

# In[4]:

from sklearn import datasets
import numpy as np
import csv
import random

#loading the cv file 
def load_data():
  data = list()
  with open('Cumulative.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    count = 0
    for row in spamreader:
      #subdata with all majors excpet math
      if row[7] != '' and row[10] != '' and row[11] != '' and row[12] != '' :
        #different feature and label for different hypothesis
        #Grade Intro Course
        #Grade Followup Course
        #Grade Fundamental Course
        sem_grad = int(row[7])
        if sem_grad > 17: sem_grad = 2
        elif sem_grad > 15 : sem_grad = 1
        else : sem_grad = 0

        grade_intro = float(row[10])
        grade_followup = float(row[11])
        grade_fundamental = float(row[12])
        
        data.append([grade_intro,grade_followup,grade_fundamental,sem_grad])
        '''
        sat_reading = int(row[4])/100
        gpa = float(row[6])
        cuml_gpa = float(row[8])
        if cuml_gpa > 3.5 : cuml_gpa = 3
        elif cuml_gpa > 3.2 : cuml_gpa = 2
        elif cuml_gpa > 2.8 : cuml_gpa = 1
        else : cuml_gpa = 0
        '''
        count = count+1
      if count == 1000 : break
    print("total sample size ", count)
  return random.sample(data,180)

a = load_data()
m = list()
n = list()
for item in a:
  m.append([item[0], item[1], item[2]])
  n.append(item[3])

X = np.array(m)
y = np.array(n)
print(type(X),'\n',X,'\n')
print(type(y),'\n',y,'\n')

'''
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print(type(X),'\n',X,'\n')
print(type(y),'\n',X,'\n')
'''

non_goal_attr1 = 'grade_intro'
non_goal_attr2 = 'avg(grade_followup, grade_fundamental)'
goal_attr1 = '0'
goal_attr2 = '1'
goal_attr3 = '2'
goal_attr4 = '3'


print('Class labels:', np.unique(y))


# Splitting data into 70% training and 30% test data:

# In[5]:


from sklearn.model_selection import train_test_split

#######################
# the four sets of data,
#	X --> features for training and testing data
#	y --> labels for training and testing data
# test_size --> proportion of the dataset to include in the test split
# random_state --> seed used by the random generator
# stratify --> y, array-like
#######################

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)


# In[6]:
print('Labels counts in y:', np.bincount(y)) 
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))


# Standardizing the features:

# In[7]:


from sklearn.preprocessing import StandardScaler

#######################
#StandardScalar --> Standardize features by removing the mean and scaling to unit variance
# transform --> Standardization by centering and scaling
#	subtracting mean and dividing with standard deviation
#######################

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    #meshgrid --> return coordinate matrices from coordinate vector
    # arange --> Return evenly spaced values within a given interval
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test samples
    #checks if test_idx range is given for test data
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')



# # Decision tree learning

# ## Maximizing information gain - getting the most bang for the buck

# In[38]:


import matplotlib.pyplot as plt
import numpy as np


def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))


def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))


def error(p):
    return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01)

ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e * 0.5 if e else None for e in ent]
err = [error(i) for i in x]

fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err], 
                          ['Entropy', 'Entropy (scaled)', 
                           'Gini Impurity', 'Misclassification Error'],
                          ['-', '-', '--', '-.'],
                          ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=5, fancybox=True, shadow=False)

ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.savefig('images/03_19.png', dpi=300, bbox_inches='tight')
plt.show()



# ## Building a decision tree

# In[39]:


from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='gini', 
                              max_depth=4, 
                              random_state=1)
tree.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, 
                      classifier=tree, test_idx=range(126, 180))

plt.xlabel(non_goal_attr1)
plt.ylabel(non_goal_attr2)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('images/03_20.png', dpi=300)
plt.show()



# In[40]:


from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

dot_data = export_graphviz(tree,
                           filled=True, 
                           rounded=True,
                           class_names=[goal_attr1, 
                                        goal_attr2,
                                        goal_attr3,
                                        goal_attr4],
                           feature_names=[non_goal_attr1, 
                                          non_goal_attr2],
                           out_file=None) 
graph = graph_from_dot_data(dot_data) 
graph.write_png('tree.png') 


# In[41]:





# ## Combining weak to strong learners via random forests

# In[42]:


from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion='gini',
                                n_estimators=25, 
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined, 
                      classifier=forest, test_idx=range(126, 180))

plt.xlabel(non_goal_attr1)
plt.ylabel(non_goal_attr2)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('images/03_22.png', dpi=300)
plt.show()



# # K-nearest neighbors - a lazy learning algorithm

# In[43]:





