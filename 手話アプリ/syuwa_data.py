import numpy as np
import pandas as pd
import os, glob
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.model_selection import KFold

from tensorflow.keras.applications import VGG16

train_dir = 'train1'

Name=[]
for file in os.listdir(train_dir):
    Name+=[file]
#print(Name)


num_classes = len(Name)
image_size = 224

N=[]
for i in range(len(Name)):
    N+=[i]
    
mapping=dict(zip(Name,N)) 
reverse_mapping=dict(zip(N,Name)) 

def mapper(value):
    return reverse_mapping[value]

#トレインセットの画像の読み込み
dataset=[]
count=0
for file in os.listdir(train_dir):
    path=os.path.join(train_dir,file)
    for im in os.listdir(path):
        image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=(224,224))
        image= np.asarray(image)
        dataset+=[[image,count]]
    count=count+1



data,labels0=zip(*dataset)
labels1=to_categorical(labels0)
y_train=np.array(labels1)
X_train=np.array(data)

kf = KFold(n_splits=5, shuffle=True)

for train_index, eval_index in kf.split(X_train,y_train):
    X_tra, X_test = X_train[train_index], X_train[eval_index]
    y_tra, y_test = y_train[train_index], y_train[eval_index]

print(X_tra.shape)
print(X_test.shape)
print(y_tra.shape)
print(y_test.shape)


"""
xy = (X_tra, X_test, y_tra, y_test)
np.save("./imagefiles_syuwa_224.npy", xy)
"""

