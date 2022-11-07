# MalConv-keras
A Keras implementation of MalConv and adversarial sample

---
## Desciprtion

This is the implementation of MalConv proposed in [Malware Detection by Eating a Whole EXE](https://arxiv.org/abs/1710.09435) which can be used for any very long sequence classification.

The adversarial samples are crafted by padding some bytes to the input file. It would fail if the origin file length exceeds the model's input size.

Enjoy !

## Requirement
- python3 (3.10.6)
- numpy (1.22.3)
- pandas (1.4.4)
- keras (2.9.0)
- tensorflow (2.9.1)
- sklearn


#### Training
```
python3 train.py DikeDataset/data_label.csv
python3 train.py DikeDataset/data_label.csv --resume
```
#### Predict
```
python3 predict.py DikeDataset/data_label.csv
python3 predict.py DikeDataset/data_label.csv --result_path saved/result.csv
```



#### Parameters
Find out more options with `-h`
```
python3 train.py -h

  -h, --help
  --batch_size BATCH_SIZE
  --verbose VERBOSE
  --epochs EPOCHS
  --limit LIMIT
  --max_len MAX_LEN
  --win_size WIN_SIZE
  --val_size VAL_SIZE
  --save_path SAVE_PATH
  --save_best
  --resume
  
python3 predict.py -h
python3 preprocess.py -h
```
#### Logs and checkpoint
The default path for output files would all be in [saved/](https://github.com/j40903272/MalConv-keras/tree/master/saved)

## Example
```
from malconv import Malconv
from preprocess import preprocess
import utils

model = Malconv()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

df = pd.read_csv(input.csv, header=None)
filenames, label = df[0].values, df[1].values
data = preprocess(filenames)
x_train, x_test, y_train, y_test = utils.train_test_split(data, label)

history = model.fit(x_train, y_train)
pred = model.predict(x_test)
```
