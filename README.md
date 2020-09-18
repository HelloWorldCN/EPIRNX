# EPIRNX #

----------
# Overview #
EPIRNX is a deep neural network based on ResNeXt used to predict enhancer-promoter interactions. Its full name is Enhancer-Promoter Interactions predictor based on ResNeXt and written in Python and implemented using TensorFlow 2.

----------
# Data Requirements #
The DNA sequence data of EPIRNX is as same as EPIANN, the details of the data and the data augmentor (Data_Augmentation.R) can be seen in [https://github.com/wgmao/EPIANN](https://github.com/wgmao/EPIANN).

As for one-hot encoding, we supply a Python scripts called sequence_processing.py, you can see it at root directory, but first, you need to use Data_Augmentation.R to get the imbalanced/balanced sequence data file which is ended with .fasta.

----------
# Train line-specific model #
You can use **train.py** to train your own models, the parameters are all have default value and explained in the following table.

| Script arguments	| Default		| Explanation					|
| ----------		| ----------	| ----------					|
| model-name		| ResNeXt		| The name of model.			|
| dataset-dir		| data/EPIs/	| Dataset root path.			|
| dataset-type		| imbalance		| balance or imbalance dataset.	|
| dataset-coding	| onehot		| Data encoding mode. onehot or embedding.		|
| cell-line			| GM12878		| The cell-line which you want to Train.	|
| batch-size		| 32			| The number of samples used to train in each epoch.	|
| epochs			| 25			| The number of training iterations.					|
| train-verbose		| 1				| The style of progress bar, 0 or 1 or 2, 0=silent, 1=progress bar, 2=one line per epoch.	|

----------
# Train transfer model #
You can use **trainsfer_train.py** to train your own trainsfer models, the parameters are all have default value and explained in the following table.

| Script arguments	| Default		| Explanation					|
| ----------		| ----------	| ----------					|
| model-name		| ResNeXt		| The name of model.			|
| dataset-dir		| data/EPIs/	| Dataset root path.			|
| dataset-type		| imbalance		| balance or imbalance dataset.	|
| dataset-coding	| onehot		| Data encoding mode. onehot or embedding.		|
| cell-line			| GM12878		| The cell-line of the pre-train model.	|
| batch-size		| 64			| The number of samples used to train in each epoch.	|
| epochs			| 20			| The number of training iterations.					|
| train-verbose		| 1				| The style of progress bar, 0 or 1 or 2, 0=silent, 1=progress bar, 2=one line per epoch.	|

----------
# Test model #
You can use **test.py** to test models, the parameters are all have default value and explained in the following table.

| Script arguments	| Default		| Explanation					|
| ----------		| ----------	| ----------					|
| model-name		| ResNeXt		| The name of model.			|
| dataset-dir		| data/EPIs/	| Dataset root path.			|
| dataset-type		| imbalance		| balance or imbalance dataset.	|
| dataset-coding	| onehot		| Data encoding mode. onehot or embedding.		|
| train-cell-line	| GM12878		| The train-cell-line of the model.				|
| batch-size		| 64			| The number of samples used to train in each epoch.	|
| test-verbose		| 1				| The style of progress bar, 0 or 1 or 2, 0=silent, 1=progress bar, 2=one line per epoch.	|


----------
# Requirements #
- Python==3.7.4
- tensorflow-gpu==2.1.0
- sklearn==0.23.2
- json==2.0.9
- numpy==1.18.5
- matplotlib==3.3.0