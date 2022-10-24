#README

##Requirements
<br>

    python-------------3.9.7
    numpy--------------1.20.3
    matplotlib---------3.4.3
    torch--------------1.10.1

<br>

##Operating Instructions
<br>
Challenge the hyperparameters in the main.py file, and then run the file
<br>
On the four datasets we used, simply tweaking the following hyperparameters yielded the results recorded in our experiments.
<br>
The main hyperparameters are as follows:
<br>
<br>

| Parameter name | Description of parameter |Recommended value |
| --- | --- |--- |
| epochs | upper epoch limit |Simple version:  CAISO: 20; Other datasets are set to 50 |
| divide_data |If divide_data=1, divide the data set according to the specific value, if divide_data=2, divide the data set according to the proportion |It is recommended that this value be set to 1. We will give the specific split length for each dataset. Due to the effect of seasonal alignment, the length of our dataset changes. In order to ensure that the length of the validation set and the test set are consistent with the baselines, this method is adopted. In fact, the data and baselines used in the training set, validation set, and test set are the same, which are 7:1:2. |
| train_len and valid_len| Hyperparameters used when divide_data=1, representing the length of training set and validation set respectively |traffic1:  11513,1755 <br> traffic2:  9494,1459 <br> electricity:  17714,2609 <br> CAISO:  17703,2632|
| horizon | The value represents h steps in the future to predict |According to the task, it can be set to 24, 48, 96, 168, of course, it can also be set to other values |
| pred_method | If the value is 1, the direct method is used, if the value is not 1, the generative method is used |Choose according to which prediction method you want to use |
<br>
Adjust the hyperparameters according to the instructions in the table, and use the default values for the rest of the hyperparameters.
<br>
<br>
You can also choose to run from the terminal using the following statement(Adjust parameters according to the table):

    python main.py --dataset traffic1 --data ./datasets/traffic1.txt --epochs 50 --divide_data 1 --train_len 11513 --valid_len 1755 --horizon 24 --pred_method 1
