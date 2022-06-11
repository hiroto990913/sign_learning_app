from django.db import models
import numpy as np
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image
import io, base64

graph = tf.compat.v1.get_default_graph()

class Photo(models.Model):
    image = models.ImageField(upload_to='photos/')
    IMAGE_SIZE = 224 #画像サイズ
    MODEL_FILE_PATH = './syuwa/ml_models/vgg16_transfer.h5' #モデル
    IMAGE_PATH = './media/photos/photo.jpg'   
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
               'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    num_classes = len(classes)
    

    def predict(self):
        model = None
        global graph
        with graph.as_default():
            
            img_data = self.image.read()
            img_bin = io.BytesIO(img_data)
            

            image = Image.open(img_bin)
            

            return image.save(self.IMAGE_PATH)

    
    def image_src(self):
        with self.image.open() as img:
            
            base64_img = base64.b64encode(img.read()).decode()

            return 'data:' + img.file.content_type + ';base64,' + base64_img


class Question(models.Model):
    subject = models.CharField(max_length=200)
    


class Predict():
    IMAGE_SIZE = 224 #画像サイズ
    MODEL_FILE_PATH = './syuwa/ml_models/vgg16_transfer.h5' #モデル
    IMAGE_PATH = './media/photos/photo.jpg'   
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N',  'O', 'P', 'Q', 'R', 'S',
               'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    def predict1(self):
        model = None
        global graph
        with graph.as_default():
            model = load_model(self.MODEL_FILE_PATH )
            image = Image.open(self.IMAGE_PATH)
            image = image.convert("RGB")
            image = image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
            data = np.asarray(image)/255.0
            
            X =[]
            X.append(data)
            X = np.array(X)

            result = model.predict([X])[0]
            predicted = result.argmax()

            return self.classes[predicted]
