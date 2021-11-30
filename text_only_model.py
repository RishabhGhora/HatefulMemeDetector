import re
import numpy as np

import tensorflow as tf
import numpy as np
import re
from tensorflow.keras.layers import Dense,Dropout,Input
from tensorflow.keras import regularizers
from transformers import *
from transformers import BertTokenizer, TFBertModel, BertConfig,TFDistilBertModel,DistilBertTokenizer,DistilBertConfig

max_len=32

def load_model():
    dbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    dbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
    inps = Input(shape = (max_len,), dtype='int64')
    masks= Input(shape = (max_len,), dtype='int64')
    dbert_layer = dbert_model(inps, attention_mask=masks)[0][:,0,:]
    dense = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(0.01))(dbert_layer)
    dropout= Dropout(0.5)(dense)
    pred = Dense(2, activation='softmax',kernel_regularizer=regularizers.l2(0.01))(dropout)
    model = tf.keras.Model(inputs=[inps,masks], outputs=pred)
    print(model.summary())
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model_save_path='dbert/dbert_model.h5'
    model.compile(loss=loss,optimizer=optimizer, metrics=[metric])
    model.load_weights(model_save_path)
    return dbert_tokenizer, dbert_model, model

def text_data_preprocess(sentences):
    chunk_words = {
      "i'm": "i am",
      "don't": "do not",
      "you're": "you are",
      "it's": "it is",
      "can't": "can not",
      "that's": "that is",
      "doesn't": "does not",
      "i'll": "i will",
      "didn't": "did not",
      "he's":"he is",
      "what's": "what is",
      "there's": "there is",
      "isn't": "is not",
      "she's": "she is",
      "let's": "let us",
      "i've": "i have",
      "they're": "they are",
      "we're": "we are",
      "ain't": "am not",
      "you've": "you have",
      "aren't": "are not",
      "you'll": "you will",
      "here's": "here is",
      "haven't": "have not",
      "i'd": "i had",
      "they'll": "they will",
      "won't": "will not",
      "who's": "who is",
      "where's": "where is",
      "couldn't": "could not",
      "shouldn't": "should not",
      "wasn't": "was not",
      "we'll": "we will",
      "idk": "i do not know",
      "y'all": "you all",
      "wife's": "wife is",
      "hasn't": "has not",
      "she'll": "she will",
      "we've": "we have",
      "they've":"they have",
      "wouldn't": "would not",
      "name's": "name is",
      "why's": "why is",
      "that'd": "that would",
      "lyin'": "lying",
      "weren't": "were not"
  }
    final_sentences = []
    for sentence in sentences:
        for key in chunk_words.keys():
            if key in sentence:
                sentence = sentence.replace(key,chunk_words[key])
        sentence = re.sub(r"'[a-z] ", ' ', sentence)
        sentence = re.sub(r"'", ' ', sentence)
        sentence = re.sub(r'[^\w\s]','',sentence)
        sentence = re.sub(' +', ' ', sentence)
        sentence = re.sub("\d+", "", sentence)
        final_sentences.append(sentence)
    return final_sentences


class Predictor():
  def __init__(self):
    dbert_tokenizer, dbert_model, model = load_model()
    self.dbert_tokenizer = dbert_tokenizer
    self.dbert_model = dbert_model
    self.model = model
    self.class_values = ["non-hateful","hateful"]
    
  def evaluate(self,text):
    preprocessed_text = text_data_preprocess(text)

    input_ids, attention_masks = [], []
    dbert_inps=self.dbert_tokenizer.encode_plus(preprocessed_text,add_special_tokens = True,max_length =max_len,pad_to_max_length = True,return_attention_mask = True,truncation=True)
    input_ids.append(dbert_inps['input_ids'])
    attention_masks.append(dbert_inps['attention_mask'])
    input_ids=np.asarray(input_ids)
    attention_masks=np.array(attention_masks)

    preds = self.model.predict([input_ids,attention_masks], batch_size=1)
    pred_label = preds.argmax(axis=1)

    return self.class_values[pred_label[0]], preds[0][pred_label[0]]