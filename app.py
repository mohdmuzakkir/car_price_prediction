from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Horesepower = int(request.form['horsepower'])
        Curb_Weight = int(request.form['curb_weight'])
        Engine_Size = int(request.form['engine_size'])
        Highway_MPG = int(request.form['highway_mpg'])
        Wheel_Base = float(request.form['wheelbase'])
        Bore = float(request.form['bore'])

        prediction=model.predict([[Horesepower, Curb_Weight, Engine_Size, Highway_MPG, Wheel_Base, Bore]])
        
        output=round(prediction[0],2)
        
        if output<0:
            return render_template('index.html',prediction_texts="Sorry you cannot sell this car")
        else:
            return render_template('index.html',prediction_text="You Can Sell The Car at $ {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
