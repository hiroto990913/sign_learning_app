import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from PIL import Image
import sys

from tensorflow.python.util.nest import _yield_sorted_items
#パラメータの初期化
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
            'L', 'M', 'N', 'Nothing', 'O', 'P', 'Q', 'R', 'S', 'Space',
           'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
num_classes = len(classes)
image_size = 40

image = Image.open(sys.argv[1])
image = image.convert("RGB")
image = image.resize((image_size, image_size))
data = np.asarray(image)/255.0
X =[]
X.append(data)
X = np.array(X)

#モデルのロード
model = load_model('./vgg16_transfer.h5')

result = model.predict([X])[0]
predicted = result.argmax()
percentage = int(result[predicted] * 100)

print(classes[predicted], percentage)