#import libraries
import numpy as np
import sklearn.datasets
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


"""# Load the model from Pickle
To load a saved model from a Pickle file, all you need to do is pass the “pickled” model into the Pickle load() function and it will be deserialized. By assigning this back to a model object, you can then run your original model’s predict() function, pass in some test data and get back an array of predictions.
"""

pickled_model = pickle.load(open('BDmodel.pkl', 'rb'))

"""# Testing the model
Detection-Whether patient has malignent or bening cancer
"""

input_data =((17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189),(13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259))

#Convert tuple into numpy array
input_data_array = np.asarray(input_data)
print(input_data_array)
print(input_data_array.shape)

#Reshape the array
input_data_reshape = input_data_array.reshape(2,-1)  #2 row, -1 indicate number of col= num of features
print(input_data_reshape)
print(input_data_reshape.shape)

#Prediction
prediction = pickled_model.predict(input_data_reshape)
print("List_Output = ",prediction) #returns a list with element [0] if malignant or [1] if bening

for pre in prediction:
  if pre == 0:
    print("Cancer is Malignant")
  else:
    print("Cancer is Bening")
