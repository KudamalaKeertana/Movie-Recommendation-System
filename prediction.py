from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the data and train the model
train = pd.read_csv("E:/Training.csv")
test = pd.read_csv("E:/Testing.csv")
train = train.drop(["Unnamed: 133"], axis=1)

P = train[["prognosis"]]
X = train.drop(["prognosis"], axis=1)
Y = test.drop(["prognosis"], axis=1)

xtrain, xtest, ytrain, ytest = train_test_split(X, P, test_size=0.2, random_state=42)
rf = RandomForestClassifier(random_state=42)
model_rf = rf.fit(xtrain, ytrain)


@app.route('/')
def home():
    return render_template('index1.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form['symptoms']
        symptoms_list = symptoms.split(',')
        input_data = np.zeros(len(X.columns))
        for symptom in symptoms_list:
            symptom_index = X.columns.get_loc(symptom)
            input_data[symptom_index] = 1
        prediction = model_rf.predict([input_data])[0]
        return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
