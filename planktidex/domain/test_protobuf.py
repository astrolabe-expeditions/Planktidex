
from keras.applications import InceptionV3
base = InceptionV3(weights="imagenet", include_top=False)

base.save('test_file.pb')

