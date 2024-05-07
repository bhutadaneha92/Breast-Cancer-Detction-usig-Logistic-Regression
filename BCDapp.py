#import libraries
import numpy as np
import sklearn.datasets
import pandas as pd
import pickle
from flask import Flask, request, render_template


#Creating Flask App
app = Flask(__name__)

#Loading the pkl file containing our trained model.
"""
# Load the model from Pickle
To load a saved model from a Pickle file, all you need to do is pass the “pickled” model into the Pickle load() function and it will be deserialized. By assigning this back to a model object, you can then run your original model’s predict() function, pass in some test data and get back an array of predictions.
"""

pickled_model = pickle.load(open('BCDmodel.pkl', 'rb'))


#Creating route for homepage
@app.route('/')
def home():
    #returns the homepage html
    return render_template('index.html')

#Creating another route for prediction
@app.route('/predict', methods = ['POST'])


# Testing the model
#Detection-Whether patient has malignent or bening cancer

def predict():
    #Get all feature values from theform
    input_features = [float(x) for x in request.form.values()]
    #Forming array of the input features
    features_values = [np.array(input_features)]
    #Since we performed feature selection on the dataset we will use the 23 features names.
    #Creating list of feature_namesinthe same order
    feature_names = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error',
       'fractal dimension error', 'worst radius', 'worst texture',
       'worst perimeter', 'worst area', 'worst smoothness',
       'worst compactness', 'worst concavity', 'worst concave points',
       'worst symmetry', 'worst fractal dimension']

    #Creating a dataframe with the input feature value and respective feature names.
    df = pd.DataFrame(feature_values, coloumns = feature_names)

    #Prediction
    prediction = pickled_model.predict(df)

    #returns a list with element [0] if malignant or [1] if bening
    #We are checking the int output storing the result as malignant or bening as a string to display op on screen.


    if prediction == [0]:
        return render_template('malignant.html')
    elif prediction == [1]:
        return render_template('benign.html')


if __name__ == "__main__":
    #Run the app
    app.run(debug = True)

    
       
