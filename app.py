from fastapi import FastAPI, File, Query, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, PlainTextResponse
import uvicorn
import joblib
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from flask_cors import CORS
import numpy as np
import bardapi
import os

# set your __Secure-1PSID value to key

app = Flask(__name__)
api = Api(app)
CORS(app)

appk = FastAPI(
    title="Credit Card Fraud Detection API",
    description="""An API that utilises a Machine Learning model that detects if a credit card transaction is fraudulent or not based on the following features: hours, amount, transaction type etc.""",
    version="1.0.0", debug=True)
class Register(Resource):
    def post(self):
        #Step 1 is to get posted data by the user
        postedData = request.get_json()
                #Get the data
        types = postedData["types"]
        amount = postedData["amount"] #"123xyz"
        oldbalanceorig=postedData["oldbalanceorig"]
        newbalanceorig=postedData["newbalanceorig"]
        oldbalancedest=postedData["oldbalancedest"]
        newbalancedest=postedData["newbalancedest"]
        isflaggedfraud=postedData["isflaggedfraud"]
        features = np.array([[1 , int(types), amount, oldbalanceorig, newbalanceorig, oldbalancedest, newbalancedest, isflaggedfraud]])
        model = joblib.load('credit_fraud.pkl')
        token = 'awhfkQ-9zUsluo6l3wUx_9HkGuU0yyeq8Fg07jSpKicVBjmjDpH2Kf6c0QKYLcIAsffWrQ.'

        # set your input text
        
        # Send an API request and get a response.
        data = {
            "types": types,
            "amount": amount,
            "oldbalanceorig": oldbalanceorig,
            "newbalanceorig": newbalanceorig,
            "oldbalancedest": oldbalancedest,
            "newbalancedest": newbalancedest,
            "isflaggedfraud": isflaggedfraud
        }
        predictions = model.predict(features)
        result=""
        if predictions == 1:
            result="fraudulent"
        elif predictions == 0:
            result="not fraudulent"
        input_text = str(data)+"and our model is predicting it as :"+str(result)
        response = bardapi.core.Bard(token).get_answer(input_text)
        print(response)
        retJson = {
            "status": 200,
            "result": result,
            "bard_response":response["content"]
        }
        return jsonify(retJson)


api.add_resource(Register, '/register')

if __name__=="__main__":
    app.run(host='0.0.0.0')
