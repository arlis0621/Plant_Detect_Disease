import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
model=load_model("")#h5 path copy
class_names=['Corn-Common_rust','Potato-Early_blight','Tomato-Bacterial_spot']

#settignt the title of the app
st.title("Plant Disease Detection")
st.markdown("Upload an image of plant leaf")

plant_image=st.file_uploader("choose an image",type="jpg")
submit=st.button("Predict")

if submit:
    if plant_image is not None:
        file_bytes=np.asarray(bytearray(plant_image.read()),dtype=np.uint8)
        opencv_image=cv2.imdecode(file_bytes,1)
        #display the image 
        st.image(opencv_image,channels='BGR')
        st.write(opencv_image.shape)
        opencv_image=cv2.resize(opencv_image,(256,256))
        opencv_image.shape=(1,256,256,3)
        y_pred=model.pedict(opencv_image)
        result=class_names[np.argmax(y_pred)]
        st.title(str(result))
        
