import keras.models

from malconv import Malconv
from preprocess import preprocess
import utils
import pandas as pd


def get_acc(y_pred, y_test):
    acc = 0

    for i in range(len(y_pred)):
        if(y_pred[i]>0.5):
            pred_label = 1
        else:
            pred_label = 0
        if(pred_label==y_test[i]):
            acc += 1
    return acc / len(y_pred)

model = Malconv()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

input_file = 'DikeDataset/data_label_short.csv'
max_len = 200000

df = pd.read_csv(input_file, header=None)
filenames, label = df[0].values, df[1].values
data = preprocess(filenames, max_len)[0]
#print(len(data))
x_train, x_test, y_train, y_test = utils.train_test_split(data, label)
print("Train set: ", len(x_train))
print("Test set: ", len(x_test))

epoch = 20
save_path = '../saved/my_train/model.h5'
prev_acc = 0

model.fit(x_train, y_train, batch_size=64)
for i in range(epoch):
    print("Epoch ", i+1)
    model = keras.models.load_model(save_path)
    model.fit(x_train, y_train, batch_size=64)
    y_pred = model.predict(x_test)
    acc = get_acc(y_pred, y_test)
    print(acc)
    if(acc>prev_acc):
        model.save(save_path)

y_pred = model.predict(x_test)
print(get_acc(y_pred, y_test))
"""
history = model.fit(x_train, y_train, epochs=2)
print(history.history)
model.save('../saved/my_train/model.h5')
y_pred = model.predict(x_test)
print(get_acc(y_pred, y_test))"""