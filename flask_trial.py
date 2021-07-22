from flask import Flask, request, render_template,get_template_attribute
import joblib
import cv2
import os
import numpy as np

app = Flask(__name__)


load_model=joblib.load('perceptron.joblib')
@app.route('/', methods =["GET", "POST"])
def image_predict():
    if request.method == "POST":
       # getting input with name = fname in HTML form
       file_name = request.form.get("myfile")
       path = os.path.join('sample', file_name)
       new_image=cv2.imread(path)[...,::-1]
       img_grey = np.mean(new_image, axis=2)                
       resized_arr = cv2.resize(img_grey, (100, 100))
       img_flatten=resized_arr.flatten()
       img_flatten=img_flatten.reshape(1,-1)
       class_pred=(load_model.predict(img_flatten)) 
       if class_pred[0]==0:
           # return render_template("pneumonia_detect_app.html",prediction="Person is healthy")
           return "Person is healthy"
       else:
           return"Person has pnuemonia"
           # return render_template("pneumonia_detect_app.html",prediction="Person has pnuemonia")
    return render_template("pneumonia_detect_app.html")

if __name__ == '__main__':

   app.run()