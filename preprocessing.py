import pandas as pd
import matplotlib.pyplot as plt
import neattext as nt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from imblearn.over_sampling import RandomOverSampler

class Preprocessing:
    def __init__(self):
        self.vocab_size=50000

    def get_vocab_size(self):
        return self.vocab_size

    def get_sentences_and_label(self,path):
        text = []
        labels = []
        f = open(path, 'r')
        for words in f:
            sentence = words.split(';')
            text.append(sentence[0])
            labels.append(sentence[1][:-1])
        return text, labels

    def text_preprocessing(self,sentences):
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.lower()
            sentence = nt.TextFrame(sentence)
            sentence.remove_emails()
            sentence.remove_urls()
            sentence = sentence.remove_emails()
            sentence = sentence.remove_emojis()
            sentence = sentence.remove_puncts()
            sentence = sentence.remove_special_characters()
            sentence = sentence.remove_stopwords()
            sentence = sentence.fix_contractions()
            cleaned_sentences.append(sentence)
        return cleaned_sentences

    def tokenizing(self,vocab_size, training_sentences, validation_sentences, testing_sentences, max_length):
        tokenizer = Tokenizer(oov_token='<OOV>', num_words=vocab_size)
        tokenizer.fit_on_texts(training_sentences)
        training_sequences = tokenizer.texts_to_sequences(training_sentences)
        training_sequences = pad_sequences(training_sequences,
                                           padding='post',
                                           truncating='post',
                                           maxlen=max_length)
        val_sequences = tokenizer.texts_to_sequences(validation_sentences)
        val_sequences = pad_sequences(val_sequences,
                                      padding='post',
                                      truncating='post',
                                      maxlen=max_length)
        testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
        testing_sequences = pad_sequences(testing_sequences,
                                          padding='post',
                                          truncating='post',
                                          maxlen=max_length)
        return tokenizer, training_sequences, val_sequences, testing_sequences

    def plot_label_distribution(self,train_labels,val_labels,test_labels):
        print("[TRAIN]")
        pd.DataFrame(train_labels).value_counts(normalize=True).plot.pie(autopct='%.2f')
        plt.ylabel("")
        plt.show()
        print("[VAL]")
        pd.DataFrame(val_labels).value_counts(normalize=True).plot.pie(autopct='%.2f')
        plt.ylabel("")
        plt.show()
        print("[TRAIN]")
        pd.DataFrame(test_labels).value_counts(normalize=True).plot.pie(autopct='%.2f')
        plt.ylabel("")
        plt.show()

    def balancing_classes(self,training_sequences,train_labels):
        ## OVERSAMPLING
        rus = RandomOverSampler(sampling_strategy='not majority')
        train_sequences_res, train_labels_res = rus.fit_resample(training_sequences, train_labels)
        print("[TRAIN]")
        pd.DataFrame(train_labels_res).value_counts(normalize=True).plot.pie(autopct='%.2f')
        plt.ylabel("")
        plt.show()

