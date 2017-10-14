import numpy as np
from keras.applications import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions


classifier = InceptionV3(weights="imagenet", include_top=True, classes=1000)


img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = classifier.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

