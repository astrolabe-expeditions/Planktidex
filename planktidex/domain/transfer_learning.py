import numpy as np
from keras.applications import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K


class PlanktonClassifier:
    N_CLASSES = 46
    
    def __init__(self, base):
        self.base = base

    def replace_last_fc_layer(self):
        cls = self.__class__
        # add a global spatial average pooling layer
        x = self.base.output
        x = GlobalAveragePooling2D()(x)
        
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(cls.N_CLASSES, activation='softmax')(x)
        model = Model(inputs=self.base.input, outputs=predictions)
        self.model = model
        return self.model

    def freeze_cnn(self):
        for layer in self.base.layers:
            layer.trainable = False
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')




class Generator:
    IMAGE_SIZE = (299, 299)
    BATCH_SIZE = 16
    
    @classmethod
    def flow_from_directory(cls, directory):
        generation_rules = cls.augmentation_generator()
        parameters = {"target_size": cls.IMAGE_SIZE,
                      "batch_size": cls.BATCH_SIZE,
                      "class_mode": "categorical"}
        generator = generation_rules.flow_from_directory(directory, **parameters)
        return generator

        
    @staticmethod
    def augmentation_generator():
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="nearest",
            zoom_range=0.3,
            width_shift_range=0.3,
            height_shift_range=0.3,
            rotation_range=360)
        return train_datagen


    

NEW_CLASS_DIRECTORY = "/home/jeanbaptiste/new_train"
DATA_AUGMENTATION = Generator.flow_from_directory(NEW_CLASS_DIRECTORY)


base = InceptionV3(weights="imagenet", include_top=False)
planktidex = PlanktonClassifier(base)
planktidex.replace_last_fc_layer()
planktidex.freeze_cnn()



# Train the model
nb_train_samples = 4125
epochs = 10
steps_per_epoch = 50
planktidex.model.fit_generator(DATA_AUGMENTATION,
                               steps_per_epoch,
                               epochs=epochs)






img_path = "/home/jeanbaptiste/new_train/diatom/9970.jpg"
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = planktidex.model.predict(x)
#print('Predicted:', decode_predictions(preds, top=3)[0])

