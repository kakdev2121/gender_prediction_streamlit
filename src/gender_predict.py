import os
import streamlit as st
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model


dir_path = os.path.dirname(os.path.realpath(__file__))
model =load_model(dir_path + "/gender_predict.h5")


alpha_number = {'x': 0,',': 1,'m': 2,'f': 3,'t': 4,'o': 5,'p': 6,'c': 7,'j': 8,'z': 9,'n': 10,'v': 11,'q': 12,'r': 13,
'a': 14,'h': 15,'e': 16,'d': 17,'k': 18,'s': 19,'g': 20,'l': 21,'i': 22,'u': 23,'w': 24,'b': 25,
'y': 26}
# max length is the maximum lenght of name in the corpus which is 15 in gender_name data base
maxlen = 15

def zero_padding(list1):
    for i in range(len(list1),maxlen):
        list1.append(0)
    return(list1)
    
def set_flag(i):
    tmp = np.zeros(len(alpha_number))
    tmp[i] = 1
    return(tmp)

def new_data_encode(name_encoded_vector):
    aa = []
    for i in name_encoded_vector:
        aa.append(set_flag(i))
    return(aa)



def gender_prediction(name):
    try:
        name = name.lower()
        chk_data = np.array(tuple(new_data_encode(zero_padding([alpha_number[i] for i in name]))))
        chk_data = chk_data.reshape(1,15,27)
        prediction = model.predict(chk_data)
        if prediction[0][0] > prediction[0][1]:
            return("Male")
        else:
            return("Female")
    except Exception as e:
            print(f"check the data {e}")


def write():
    st.header("Gender Predictor")
    input_name = st.text_input('Input First name')
    if input_name:
        gender = gender_prediction(input_name)
        st.success(gender)
    else:
        st.error('Wrong Input')
