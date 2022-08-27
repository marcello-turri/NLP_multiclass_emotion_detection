import keras.models
import numpy as np
from preprocessing import Preprocessing
from sklearn.preprocessing import LabelEncoder
from model import Model

# IMPORTING THE DATASET
train_path = 'train.txt'
val_path = 'val.txt'
test_path = 'test.txt'
preprocessing = Preprocessing()
train_sentences, train_labels = preprocessing.get_sentences_and_label(train_path)
val_sentences, val_labels = preprocessing.get_sentences_and_label(val_path)
test_sentences, test_labels = preprocessing.get_sentences_and_label(test_path)

sent_train_lens = [len(sentence.split()) for sentence in train_sentences]
avg_sent_train_len = np.mean(sent_train_lens)
sent_val_lens = [len(sentence.split()) for sentence in val_sentences]
avg_sent_val_len = np.mean(sent_val_lens)
sent_test_lens = [len(sentence.split()) for sentence in test_sentences]
avg_sent_test_len = np.mean(sent_test_lens)

print(f"train {avg_sent_train_len}")
print(f"val {avg_sent_val_len}")
print(f"test {avg_sent_test_len}")


train_sentences = preprocessing.text_preprocessing(train_sentences)
val_sentences = preprocessing.text_preprocessing(val_sentences)
test_sentences = preprocessing.text_preprocessing(test_sentences)



tokenizer,training_sequences,val_sequences,testing_sequences = preprocessing.tokenizing(preprocessing.vocab_size,
                                                                                        train_sentences,
                                                                                        val_sentences,
                                                                                        test_sentences,
                                                                                        max_length=int(avg_sent_train_len)+1)
preprocessing.plot_label_distribution(train_labels,val_labels,test_labels)
le = LabelEncoder()
model = Model()
model.compile()
history = model.fit(training_sequences,
                    le.fit_transform(train_labels),
                    val_sequences,
                    le.fit_transform(val_labels))
model.plot_curves(history)
model = keras.models.load_model("saved_model/my_model.h5")
predictions = model.predict(testing_sequences)
predictions = np.argmax(predictions,axis=1)
accuracy,precision,recall,f1 = model.evaluate(le.fit_transform(test_labels),predictions,cm=True)
print(accuracy,precision,recall,f1)
model.save('saved_model/my_model.h5')

