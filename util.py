import cv2
import numpy as np
import base64
from server.wavelet import WtoD

import json
import joblib
__class_name_to_number = {}
__class_number_to_name = {}
__model = None


def classify_image(image_b64, file_path= None):
    imgs = get_cropped_image(file_path,image_b64)
    result = []
    for img in imgs:
        print(img.dtype)
        scaled_raw_img = cv2.resize(img, (32,32))
        img_har = WtoD(scaled_raw_img,'db1',5)
        scaled_img_har = cv2.resize(img_har, (32,32))
        combined_img =np.vstack((scaled_raw_img.reshape(32*32*3,1),scaled_img_har.reshape(32*32,1)))
        len_img_array = 32*32*3+32*32
        final = combined_img.reshape(1,len_img_array).astype(float)
        print("__model.predict(final)",type(__model.predict(final)))
        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_prob': np.round(__model.predict_proba(final)*100,2).tolist()[0],
            'class_dicitonary': __class_name_to_number
            })
    return result
        
def load_saved_artifacts():
    print("loading artifacts")
    global __class_name_to_number
    global __class_number_to_name
    
    with open("./server/artifacts/class_dict.json","r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}
    
    global __model
    if __model is None:
        with open("./server/artifacts/saved_model.pkl","rb") as f:
            __model = joblib.load(f)
    print("loaded artifacts")

def class_number_to_name(class_num):
   return __class_number_to_name[class_num]

def get_cv2_image_from_base_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data),np.uint8)
    img =  cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img
    
def get_cropped_image(image_path, image_b64):       
    face_cascade = cv2.CascadeClassifier(r'./model/opencv/haarcascade/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(r'./model/opencv/haarcascade/haarcascade_eye.xml')
    
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base_string(image_b64)
    
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey,1.3,5)
    cropped_Faces = []
    for(x,y,w,h) in faces:
        roi_gray = grey[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if(len(eyes) >=2):
            cropped_Faces.append(roi_color)
    return cropped_Faces

def get_b64_test_image_for_jeff():
    with open("./server/test_image/jeff.txt") as f:
        return f.read()

if __name__=="__main__":
    load_saved_artifacts()
    print(classify_image(get_b64_test_image_for_jeff(), None))
    