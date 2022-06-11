import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import  accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import  Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold

train_dir = 'train1'
test_dir = 'test1'

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


#テストセットの画像の読み込み
testset=[]
count=0
for file in os.listdir(test_dir):
    path=os.path.join(test_dir,file)
    for im in os.listdir(path):
        image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=(224,224))
        image=img_to_array(image)
        image=image/255.0
        testset+=[[image,count]]
    count=count+1

test,testlabels0=zip(*testset)
labels1=to_categorical(testlabels0)
labels=np.array(labels1)
test=np.array(test)

#データの読み込み
X_train, X_test, y_train, y_test = np.load('./imagefiles_syuwa_224.npy',allow_pickle=True)

X_train = X_train.astype("float") / 255.0
X_test = X_test.astype("float") / 255.0

#print(y_train.shape)
#print(y_test.shape)


#モデルの定義
model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size,image_size,3))


top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))#入力したモデルを直列に並べる
top_model.add(Dense(256, activation='relu'))#全結合層を追加する
top_model.add(Dropout(0.5))#半分データを捨てる
top_model.add(Dense(num_classes, activation='softmax'))#最終的な出力

model = Model(inputs=model.input, outputs=top_model(model.output))




for layer in model.layers[:15]:
    layer.trainable = False
model.summary()

'''
opt = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=15,validation_data=(X_test,y_test))

model.save("./aiapps/syuwa/ml_models/vgg16_transfer.h5")

score = model.evaluate(test, labels, batch_size=32)
'''
