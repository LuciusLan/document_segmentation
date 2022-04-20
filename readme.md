## Dependencies:
torch>=1.8.1

numpy>=1.20.1

pandas>=1.3.5

scipy>=1.6.2

huggingface>=4.11.3

---
## Before runing the codes

Download the competetion dataset from Kaggle: https://www.kaggle.com/competitions/feedback-prize-2021/data

(If kaggle CLI tool is available, run `> kaggle competitions download -c feedback-prize-2021` to download)


Move the dataset zip file to the project SourceCode folder, and unzip.
The file structure should appear as following:

```
SourceCode/
|
├─── train/
|    ├── ********.txt
|    ├── ********.txt  
|    ├── ...
|    ├── ...
|    |
|    
├─── test/
|    ├── ********.txt
|    ├── ********.txt  
|    ├── ...
|    ├── ...
|
├── sample_submission.csv
├── train.csv
|
|
├── main.py
├── model.py
├── data.py
├── params.py
└── forKaggles.ipynb

```

Due to the nature of the model used, if CUDA device is not available, the training would take too long time and hence usage of CUDA device is set as default setting, training on CPU is not supported.

One may change in `params.py` to set different hyperparameters. The last line of `params.py` needs to be un-commented; set `MODEL_CACHE_DIR = None` to let downloading pretrained transformer model to default directory; or set a path to a desired cached directory; 

Explanations of other parameters:

`TEST_SIZE = 0`

This line specifies the size to split a testing subset from the training data. Default to zero as Kaggle has its own private testing set.

`DEV_SIZE = 0.1`

This line specifies the size to split a validating subset from the training data. Default to be 10% of training data.

`MAX_LEN = 512`

This line specifies the maximum sequence length of the transformer model can take. Sliding window creation is based on this setting. 512 for RoBERTa model. For Longformer model, although it can take input sequence up to 4096 tokens, we used 1024 in our experiment due to the GPU memory limit.

`BATCH_SIZE = 4`

To specify the batch size for training

`LEARNING_RATE = 2e-5`

To specify the learning rate

`NUM_EPOCH = 10`

To specify the number of epochs to train

`LSTM_HIDDEN = 300`

To specify the hidden dimension for LSTMs used

`BIAFFINE_DROPOUT = 0.3`

To specify the dropout rate for LSTMs used

`BASELINE = False`

Set to True to run in baseline setting (Transformer + one simple feed-forward layer)

`LONGBERT = False`

Set to True to use Longformer instead of RoBERTa for the transformer backbone


---

## Training and Testing


`> cd` to the `SourceCode` folder to run the training and testing.

Run `SourceCode> python main.py` to start training and testing.

Do note that `forKaggles.ipynb` is **NOT** for local training and testing. The notebook file is edited for submission to Kaggle, and can only be executed in Kaggle's enviornment. We omit the instructions for make submission to Kaggle here.

The output file will be generated after training is completed, written to `output.csv`.

The format of output file is following that of Kaggle's requirements, in the format of the following:

```
id,class,predictionstring
1,Claim,1 2 3 4 5
1,Claim,6 7 8
1,Claim,21 22 23 24 25
```

Where 'id' column will contain the document ids (file names) of test .txt files, 'class' column contains the predicted class, and 'predictionstring' column contains the predicted span of segements, recording the posisitions of words inside the segment, separated by whitespaces.