#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib import gridspec 
data = pd.read_csv(r"C:\Users\praanya bhatnagar\Downloads\creditcard.csv")
#data.head()               fetching the data form the file csv
#print(data.shape) 
#print(data.describe()) 
# fraud cases in dataset 
fraud = data[data['Class'] == 1] 
valid = data[data['Class'] == 0] 
outlierFraction = len(fraud)/float(len(valid)) 
print(outlierFraction) 
fraud_count=len(data[data['Class'] == 1])
valid_count=len(data[data['Class'] == 0])
print('Fraud Cases: {}',fraud_count) 
print('Valid Transactions: {}',valid_count) 
print('\nAmount details of the fraudulent transaction') 
fraud.Amount.describe() 
print('\ndetails of valid transaction') 
valid.Amount.describe() 


# In[30]:


# Correlation matrix plotting by sns
corrmat = data.corr() 
fig = plt.figure(figsize = (12, 9)) 
sns.heatmap(corrmat, vmax = .8, square = True) 
plt.show() 
X = data.drop(['Class'], axis = 1) 
Y = data["Class"] 
print(X.shape) 
print(Y.shape)
xData = X.values 
yData = Y.values 


# In[23]:


# Using Scikit-learn to split data into training and testing sets 
from sklearn.model_selection import train_test_split 
xTrain, xTest, yTrain, yTest = train_test_split( 
		xData, yData, test_size = 0.2, random_state = 42) 
#  Forest Classifier 
from sklearn.ensemble import RandomForestClassifier 
rfc = RandomForestClassifier() 
rfc.fit(xTrain, yTrain) 
# predictions 
yPred = rfc.predict(xTest) 
from sklearn.metrics import classification_report, accuracy_score  
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import f1_score, matthews_corrcoef 
from sklearn.metrics import confusion_matrix 
  
n_outliers = len(fraud) 
n_errors = (yPred != yTest).sum() 
print("The model used is Random Forest classifier") 
  
acc = accuracy_score(yTest, yPred) #measure of how well the classifier correctly predicts both classes
print("The accuracy is {}".format(acc)) 
  
prec = precision_score(yTest, yPred) #ratio of correctly predicted positive observations to the total predicted positives. 
print("The precision is {}".format(prec)) 
  
rec = recall_score(yTest, yPred) #sensitivity or true positive rate,
print("The recall is {}".format(rec)) 
  
f1 = f1_score(yTest, yPred) #weighted average of precision and recall
print("The F1-Score is {}".format(f1)) 
  
MCC = matthews_corrcoef(yTest, yPred) #measure of the quality of binary and multiclass classifications
print("The Matthews correlation coefficient is{}".format(MCC)) 


# In[25]:


# printing the confusion matrix 
#showing the number of true positive, true negative, false positive, and false negative predictions.
LABELS = ['Normal', 'Fraud'] 
conf_matrix = confusion_matrix(yTest, yPred) 
plt.figure(figsize =(12, 12)) 
sns.heatmap(conf_matrix, xticklabels = LABELS, 
			yticklabels = LABELS, annot = True, fmt ="d"); 
plt.title("Confusion Matrix") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show() 


# In[ ]:




