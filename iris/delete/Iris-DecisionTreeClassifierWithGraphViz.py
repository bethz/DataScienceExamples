
# coding: utf-8

# In[26]:


from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import tree
import numpy as np


# ## Iris is the Hello World for Machine Learning
# it is packaged in scikit-learn
# info on the data is available from https://scikit-learn.org/stable/datasets/index.html#iris-dataset
# 
# 
# This is a very basic intro.

# In[27]:


# Load scikit-learn data
iris = datasets.load_iris()


# In[28]:


# Create Training target (aka iris name) and data 
# Remove Test data from set
#
# Remove one row of data for each type of iris for use in testing our prediction later.
test_index = [0,50,100]
train_target = np.delete(iris.target, test_index)
# check axis=0 for deleting row
train_data = np.delete(iris.data, test_index, axis=0)


# In[29]:


# Create Test target and data
test_target = iris.target[test_index]
test_data= iris.data[test_index]


# In[30]:


# Create Decision Tree Classifier
clf = tree.DecisionTreeClassifier()


# In[31]:


pred = clf.fit(train_data,train_target)
score = clf.predict(test_data)
# print the expected predictions
print("Expected predictions :",test_target)
print("Actual predictions   :",score)


# In[43]:


get_ipython().system('pip install pydot')
get_ipython().system('pip install pydotplus')
get_ipython().system('pip install pygraphviz')


# In[44]:


import pydotplus
import Graphviz


# In[46]:


#viz code
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf,
                    out_file=dot_data,
                    feature_names=iris.feature_names,
                    class_names=iris.target_names,
                    filled=True, rounded=True,
                    impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
print(graph)
graph.write_pdf("iris.pdf")

