import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import os
import time
import cv2

import numpy as np
#load the trained model to classify sign
from keras.models import load_model
model = load_model('traffic_classifier.h5')

#dictionary to label all traffic signs class.
classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',      
            3:'Speed limit (50km/h)',       
            4:'Speed limit (60km/h)',      
            5:'Speed limit (70km/h)',    
            6:'Speed limit (80km/h)',      
            7:'End of speed limit (80km/h)',     
            8:'Speed limit (100km/h)',    
            9:'Speed limit (120km/h)',     
           10:'No passing',   
           11:'No passing veh over 3.5 tons',     
           12:'Right-of-way at intersection',     
           13:'Priority road',    
           14:'Yield',     
           15:'Stop',       
           16:'No vehicles',       
           17:'Veh > 3.5 tons prohibited',       
           18:'No entry',       
           19:'General caution',     
           20:'Dangerous curve left',      
           21:'Dangerous curve right',   
           22:'Double curve',      
           23:'Bumpy road',     
           24:'Slippery road',       
           25:'Road narrows on the right',  
           26:'Road work',    
           27:'Traffic signals',      
           28:'Pedestrians',     
           29:'Children crossing',     
           30:'Bicycles crossing',       
           31:'Beware of ice/snow',
           32:'Wild animals crossing',      
           33:'End speed + passing limits',      
           34:'Turn right ahead',     
           35:'Turn left ahead',       
           36:'Ahead only',      
           37:'Go straight or right',      
           38:'Go straight or left',      
           39:'Keep right',     
           40:'Keep left',      
           41:'Roundabout mandatory',     
           42:'End of no passing',      
           43:'End no passing veh > 3.5 tons' }
                 
#initialise GUI
top=tk.Tk() #where top is the name of the main window object

# setting to full screen
width= 0.8*top.winfo_screenwidth()               
height= 0.8*top.winfo_screenheight()               
top.geometry("%dx%d+100+50" % (width, height))

#title bar
top.title('MDP GAN Traffic Sign Classifier') 
top.configure(background='#f1faee') #done

label=Label(top,background='#f1faee', font=('Poppins',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((30,30))
    numpy1 = np
    image = numpy1.expand_dims(image, axis=0)
    image = numpy1.array(image)
    print(image.shape)

    if len(image.shape) > 2 and image.shape[3] == 4:
        print(image.shape)
        #convert the image from RGBA2RGB
        # image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        image = image[:, :, :, :3]
        
    
    print(image.shape)

    # pred = model.predict_classes([image])[0]

    predict_x = model.predict([image])[0]
    pred = np.argmax(predict_x, axis=0)

    sign = classes[pred+1]
    print(sign)
    label.configure(foreground='#1d3557', text=sign, font=('Poppins',20, 'bold')) 

def run_superres(file_path):
    head, tail = os.path.split(file_path)
    cmstring = "super_res_cnn -i test/"+tail+" -o output.png"
    os.system(cmstring)
    # time.sleep(5.2)
    newfp = 'output.png'
    uploaded=Image.open(newfp)
    uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
    im=ImageTk.PhotoImage(uploaded)

    sign_image.configure(image=im)
    sign_image.image=im
    label.configure(text='')

    show_classify_button(file_path)
# C:/Users/navan/OneDrive/Desktop/02 - Projects/MDP/project/portable tool/
# realesrgan-ncnn-vulkan -i test/001.png -o output.png
# "realesrgan-ncnn-vulkan -i input.png -o output.png"
    pass


def show_superres_button(file_path):
    superres_b=Button(top,text="Enhance image",command=lambda: run_superres(file_path),padx=10,pady=5)
    superres_b.configure(background='#2ec4b6', foreground='white',font=('Poppins',14, 'bold'))
    superres_b.place(relx=0.79,rely=0.36)

def show_classify_button(file_path):
    classify_b=Button(top,text="What is the sign?",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#e63946', foreground='white',font=('Poppins',14, 'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        # print(file_path)
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_superres_button(file_path)
        # show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#e63946', foreground='white',font=('Poppins',14,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Vehicle Dashboard Traffic Sign Detector",pady=20, font=('Poppins',20,'bold'))
heading.configure(background='#f1faee',foreground='#1d3557')
heading.pack()

top.mainloop() #infinite loop used to run the application, wait for an event to occur and process the event as long as the window is not closed.
