import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, classification_report
import PIL 
import os 
import numpy as np
import seaborn as sns

traindir = 'train'
testdir = 'test'
batch_size = 64

class XrayClassifier:
    def __init__(self, traindir, testdir, batch_size):
        self.traindir = traindir
        self.testdir = testdir
        self.batch_size = batch_size
        self.model = self.build_model()
        self.history = None 

    def create_data_generators(self):
        training_generator = ImageDataGenerator(
            rescale=1.0/255,
            zoom_range=0.1,
            rotation_range=25,
            width_shift_range=0.05,
            height_shift_range=0.05
        )
        validation_generator = ImageDataGenerator()

        self.training_iterator = training_generator.flow_from_directory(
            self.traindir,
            class_mode='categorical',
            color_mode='grayscale',
            batch_size=self.batch_size
        )

        self.testing_iterator = validation_generator.flow_from_directory(
            self.testdir,
            class_mode='categorical',
            color_mode='grayscale',
            batch_size=self.batch_size
        )

    def build_model(self):
        self.model = Sequential()
        #tf.random.set_seed(42)
        self.model.add(layers.Conv2D(8, (3, 3), activation="relu", input_shape=(256, 256, 1)))
        self.model.add(layers.MaxPooling2D(2, 2))
        self.model.add(layers.Conv2D(16, (3, 3), activation="relu"))
        self.model.add(layers.MaxPooling2D(2, 2))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(16, activation='relu'))
        self.model.add(layers.Dense(3, activation='softmax'))

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )

    def train_model(self, epochs=10):
    	#Early stopping to prevent accuracy from plateau/overfitting
    	es = EarlyStopping(monitor='accuracy', mode='min', verbose=1, patience=20)
    	self.history = self.model.fit(
            self.training_iterator,
            epochs=epochs,
            steps_per_epoch=self.training_iterator.samples // self.batch_size,
            validation_data=self.testing_iterator,
            validation_steps=self.testing_iterator.samples // self.batch_size,
            callbacks=[es]
        )

    def evaluate_model(self):
    	loss, accuracy = self.model.evaluate(self.testing_iterator)
    	print("Model Loss:", loss)
    	print("Model Accuracy:", accuracy)
    	prediction = self.model.predict(self.testing_iterator)
    	labels = []
    	for _ in range(len(self.testing_iterator)):
    		_, label = self.testing_iterator.next()
    		labels.extend(label)
    	r2 = r2_score(labels, prediction)
    	#print("R-Squared:", r2)

    def predict(self, image):
        img = load_img(image, target_size=(256,256), color_mode='grayscale')
        image_array = img_to_array(img)
        image_array = image_array / 255
        image_array = np.expand_dims(image_array, axis=0)
        prediction = self.model.predict(image_array)
        pclass = np.argmax(prediction)
        if pclass == 0:
            print("This looks like Covid-19")
        elif pclass == 1:
            print("Perfectly normal")
        else:
            print("Seems like Pneumonia")

    def plot_epoch_accuracy(self):
    	plt.figure(figsize=(10,6))
    	plt.plot(range(1,11),self.history.history['accuracy'], label='Training Accuracy')
    	plt.plot(range(1,11),self.history.history['val_accuracy'], label='Validation Accuracy')
    	plt.title('Model Accuracy')
    	plt.xlabel('Epoch')
    	plt.xticks(np.arange(1, 11, 1))
    	plt.ylabel('Accuracy')
    	plt.legend()
    	plt.show()

    def plot_epoch_loss(self):
    	plt.figure(figsize=(10,6))
    	plt.plot(range(1,11),self.history.history['loss'], label='Training Loss')
    	plt.plot(range(1,11),self.history.history['val_loss'], label='Validation Loss')
    	plt.title('Training and Validation Loss')
    	plt.xlabel('Epoch')
    	plt.ylabel('Loss')
    	plt.xticks(np.arange(1, 11, 1))
    	plt.legend()
    	plt.show()

    def plot_prediction_actual(self):
    	predictions = []
    	actualvals = []
    	plt.figure(figsize=(8,8))
    	for _ in  range(66):
    		test, actual = self.testing_iterator.next()
    		actualvals.extend(np.argmax(actual, axis=1))

    		prediction = self.model.predict(test)
    		predictions.extend(np.argmax(prediction, axis=1))
    	cmatrix = confusion_matrix(actualvals, predictions)

    	sns.heatmap(cmatrix, annot=True, fmt='d', cmap='Blues')
    	plt.xlabel('Actual Classification')
    	plt.ylabel('Predicted Classification')
    	plt.title('Confusion Matrix')
    	plt.show()



image = 'testpredict.png'

model = XrayClassifier(traindir, testdir, batch_size)
model.create_data_generators()
model.build_model()
model.train_model()
model.evaluate_model()
model.predict(image)
model.plot_epoch_accuracy()
model.plot_epoch_loss()
model.plot_prediction_actual()

