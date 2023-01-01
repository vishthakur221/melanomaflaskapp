from flask import Flask, render_template, request
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions

loaded_model_imageNet=load_model("vishal_model_resnet50.h5")


app = Flask(__name__)

# loaded_model_imageNet=load_model("vishal_model_resnet50.h5")

# UPLOAD_FOLDER = "C:/Users/hp/Desktop/Flask_Project_Melanoma/static/images"
UPLOAD_FOLDER = "static/images"

@app.route('/', methods=['GET'])
def hello_word():
    return render_template("index.html", imageExist=True)

@app.route('/', methods=['POST'])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["patientImage"]
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location)
            # ------------------New Code-----------------------------
            img_path=image_location
            img=cv2.imread(img_path)
            img=cv2.resize(img,(100,100))
            x=np.expand_dims(img,axis=0)
            x=preprocess_input(x)
            result=loaded_model_imageNet.predict(x)
            p=list((result*100).astype('int'))
            pp=list(p[0])
            ss = max(pp)
            index=pp.index(max(pp))
            name_class=['Benign','Malignant']
            Final_Result = name_class[index]
            # ------------------New Code End-------------------------
            return render_template("index.html", result = Final_Result, imageName= image_file.filename, imageExist=False)
        # return render_template("dummy.html", file=image_file)






if __name__ == '__main__':
    # loaded_model_imageNet=load_model("/home/vishthakur/mysite/vishal_model_resnet50.h5")
    app.run(debug=True)
    # app.run(port=3000, debug=True)
