from flask import Flask, jsonify, request, abort
from Service.ModelService import ModelService
import json

app = Flask(__name__)

service = ModelService(chosen_classifier="default")

@app.route('/predict', methods=['POST'])
def predict():
    prediciton = service.predict(request.json)
    return json.dumps({ 'result': prediciton})

@app.route('/report', methods=['GET'])
def report():
    retval = service.modelReport()
    return json.dumps(retval)


if __name__ == '__main__':
    app.run()
