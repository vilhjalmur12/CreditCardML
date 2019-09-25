# Machine learning credit card fraud detection
## Overview
This is an API implementation of a supervised machine learning program to detect frauds on credit card
usage. The dataset is synthetic and only made for research purposes. 

The program is equipped with algorithms such as K-nearest neighbour, Support vector machines, Random forest,
decision trees and an artificial neural network. Although what is actually used is the 
support vector machine. The reason for that is that I have not had the time to test hyperparameters
for other algorithms such as random forest and neural network but it is a future project. The SVM still
produces great result from the dataset.

## Usage
Assuming all packages have been installed including python3 and flask. Also the best 
tool to send requests would be postman.

Inside the static folder in the root of the project you can find a JSON object. This object is how a new
transaction would look like when you post it to the server.

1. run the app.py file to run on your localhost on port 5000
2. to get the report on the precision of the model run in postman a GET method: 
    http://localhost:5000/report 
3. to get a result of 1 or more transaction predictions run in postman a POST method:
    http://localhost:5000/predict with a JSON body of the object you find in the static folder.
   
