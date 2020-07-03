# 1.DOWNLOAD DATA

#导入模块
import urllib.request
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tarfile

#下载数据集
url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath="/Users/wenyujin/.keras/datasets/aclImdb_v1.tar.gz"
if not os.path.isfile(filepath):
    result=urllib.request.urlretrieve(url,filepath)

#解压压缩文件
if not os.path.exists("/Users/wenyujin/.keras/datasets/aclImdb"):
    tfile = tarfile.open("/Users/wenyujin/.keras/datasets/aclImdb_v1.tar.gz", 'r:gz')
    result=tfile.extractall('/Users/wenyujin/.keras/datasets/')

#2.DATA PROCESSSING(CLEAN AND TOKENIZE)

#CLEAN

#导入所需模块
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

#删除文字中的HTML标签
import re
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)

#读取IMDb文件目录
import os
def read_files(filetype):
    path="/Users/wenyujin/.keras/datasets/aclImdb/"
    file_list=[]
    positive_path= path+ filetype+ "/pos/"
    for f in os.listdir(positive_path):
        file_list+=[positive_path+f]
    negative_path= path+ filetype+ "/neg/"
    for f in os.listdir(negative_path):
        file_list+=[negative_path+f]


    print('read',filetype, 'files:',len(file_list))
    all_labels= ([1] * 12500 + [0] * 12500)
    all_texts= []
    for fi in file_list:
        with open(fi,encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]
    return all_labels,all_texts

#读取数据目录
y_train,train_text=read_files("train")
y_test,test_text=read_files("test")
print(train_text[0])
print(y_train[0])
print(train_text[12500])
print(y_train[12500])

#TOKENIZE

#建立token字典
token = Tokenizer(num_words=2000)
token.fit_on_texts(train_text)
print(token.document_count)

#转化为数字列表
x_train_seq= token.texts_to_sequences(train_text)
x_test_seq= token.texts_to_sequences(test_text)
print(x_test_seq[0])

#Padding
x_train= sequence.pad_sequences(x_train_seq,maxlen=150)
x_test= sequence.pad_sequences(x_test_seq,maxlen=150)

print('before pad_sequence length=',len(x_train_seq[0]))
print(x_train_seq[0])
print('after pad_sequence length=',len(x_train[0]))
print(x_train[0])

print('before pad_sequence length=',len(x_test_seq[0]))
print(x_test_seq[0])
print('after pad_sequence length=',len(x_test[0]))
print(x_test[0])

#EMBEDDING

#导入所需模块
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM

#建立model
model = Sequential()
model.add(Embedding(output_dim=32,
                    input_dim=2000,
                    input_length=150))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

# Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=100,
    validation_data=(x_test,y_test),
    validation_split=0.2
)

# Evaluate the accuracy of model
scores = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', scores[1])

