#import libraries
import numpy as np
import sklearn.datasets
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""0- malignant
1- benign

# Load Dataset from scikit learn
"""

dataset = sklearn.datasets.load_breast_cancer()
#print(dataset)

"""# Read the dataset"""

X = dataset.data
Y = dataset.target
#print(f"X={X}\n Y={Y}")
print(f"X={X.shape}\n Y={Y.shape}")

#Create a dataframe
data = pd.DataFrame(dataset.data, columns = dataset.feature_names)
data['class'] = dataset.target
data.head()

#data.describe()

#counts how many malignant i.e. 1 and bening i.e. 0 are present
#print(data['class'].value_counts())
#print(dataset.target_names)
data.groupby('class').mean() #print the mean value of targets

"""# Splitting of Data"""

#Split data into train and test

#####X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1) #Size of test data is 10%
#print(Y.shape, Y_train.shape, Y_test.shape)

####To see how the data is split.
####If the mean values are close to each other then data splitted equally else data distribution is unequal
#print("Malignent and benign cases are unequally distributed=",Y.mean(), Y_train.mean(), Y_test.mean())

#####We need to used stratify parameter to split the data equally for target class
#X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1, stratify = Y)
#print("Malignent and benign cases are equally distributed=",Y.mean(), Y_train.mean(), Y_test.mean())

#If we run the code multiple times i.e. Reproduce the code, data should be split in a same manner as before to achieve same results
#Random state is used to split the data in a specific manner
#each value of random_state splits the data differently
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1, stratify = Y, random_state=1)
print("splitting of data with random_state=",Y.mean(), Y_train.mean(), Y_test.mean())

"""# Define and train the model
Logistic regression used for binary classification. It uses sigmoid function

Input is continuous
output is categorial i.e. 1 and 0
"""

classifier = LogisticRegression(random_state=0) #Loadig the model to the variable classifier

#Traing the model
classifier.fit(X_train, Y_train)

"""#Evaluation of model"""

#Accuracy for training data
Train_predicted = classifier.predict(X_train) #Prediction class labels for sample in X_train
Train_accuracy = accuracy_score(Y_train,Train_predicted)
print("Accuracy for training data=",Train_accuracy)

#Accuracy for testing data
Test_predicted = classifier.predict(X_test) #Prediction class labels for sample in X_test
Test_accuracy = accuracy_score(Y_test,Test_predicted)
print("Accuracy for testing data=",Test_accuracy)

"""# Save the model with Pickle
To save the ML model using Pickle all we need to do is pass the model object into the dump() function of Pickle. This will serialize the object and convert it into a “byte stream” that we can save as a file called model.pkl. You can then store, or commit to Git, this model and run it on unseen test data without the need to re-train the model again from scratch.
"""

pickle.dump(classifier, open('BCDmodel.pkl', 'wb'))

