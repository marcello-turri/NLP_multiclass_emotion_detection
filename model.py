from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics._classification import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
import tensorflow as tf
from keras import regularizers
from preprocessing import Preprocessing
import matplotlib.pyplot as plt

class Model:
    def __init__(self):
        self.input_dim = Preprocessing().get_vocab_size()
        self.output_dim = 64
        self.input_length = 20
        self.model_ = self.build_model()
        self.callbacks = [
            EarlyStopping(patience=10, monitor='val_loss',restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss',min_lr=1e-7,patience=2,mode='min',factor=0.1),
            ModelCheckpoint(monitor='val_loss',filepath='checkpoint',save_best_only=True)
        ]

    def build_model(self):
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

    def fit(self, x, y,val_x,val_y):
        return self.model_.fit(x, y,epochs=20,
                               validation_data=(val_x,val_y),
                               callbacks=[self.callbacks])

    def predict(self, x):
        predictions = self.model_.predict(x)
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
            plt.show()

        return accuracy, precision, recall, f1

    def save(self,path):
        self.model_.save(path)
