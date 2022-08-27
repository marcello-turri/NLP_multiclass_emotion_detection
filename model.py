from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics._classification import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import tensorflow as tf
from keras import regularizers
from preprocessing import Preprocessing
import matplotlib.pyplot as plt

class Model:
    def __init__(self):
        self.input_dim = Preprocessing().getVocabSize()
        self.output_dim = 64
        self.input_length = 20
        self.model_ = self.buildModel()

    def buildModel(self):
        model_ = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=50000,
                                      output_dim=64,
                                      input_length=20),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.LSTM(128, return_sequences=False),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(6, activation='softmax')
        ])
        return model_

    def compile(self):
        self.model_.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    def fit(self, X, y,valX,valY):
        return self.model_.fit(X, y,epochs=15,
                    validation_data=(valX,valY))

    def predict(self, X):
        predictions = self.model_.predict(X)
        return predictions

    def plot_curves(self,history):
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        plt.plot(train_acc)
        plt.plot(val_acc)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    def evaluate(self,ytrue, ypred, cm=False):
        accuracy = accuracy_score(ytrue, ypred)
        precision, recall, f1, _ = precision_recall_fscore_support(ytrue,
                                                                   ypred,
                                                                   average='weighted')
        if cm == True:
            conf_matrix = confusion_matrix(ytrue, ypred)
            disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
            disp.plot(cmap=plt.cm.Blues)

        return accuracy, precision, recall, f1
