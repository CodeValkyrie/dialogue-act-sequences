# Language Modelling of Dialogue Act Sequences

This project evaluates the prediction performance of different models on Dialogue Act (DA) prediction given sequences of DAs. The models include baseline models (majority class, random class, weighted random class, bigram and trigram) and two LSTM models. The first LSTM model takes in the input features (speaker, DA, level, utterance length) and is called the 'old' model. The old model has a weighted and unweighted version. The second LSTM model has the same input features as the old model but with added text embedding. This is the 'new' model.

## Getting Started

To get this repository working, several python libraries need to be installed.

### Dependencies

numpy (1.16.5)
pandas (0.25.1)
pytorch (1.3.1)
json (2.0.9)
seaborn (0.9.0)
matplotlib (3.1.1)
scipy (1.3.1)
sklearn (0.21.3)
nltk (3.4.5)

## Directories and Files

### Directories

```
analyses    - will be created when running the results scripts; contains all the result tables and graphs.
data        - contains the data set and the train/test split. Will later contain preprocessed data files.
remnants    - contains all the files that were used, but are not used any more.
```


### Class Files

These files contain the classes that are used in the scripts.

```
cross_validation_new_model.py
cross_validation_old_model.py
data.py
new_model_with_text.py
old_model_without_text.py
train.py
```


### Scripts

These files are explained in the next section.

```
python script_baselines.py
python script_error_analysis.py
python script_graphs_results
python script_graphs_statistics
python script_n_gram_models.py
python script_new_model_hyperparameter_search.py
python script_new_model_precision_recall_f1.py
python script_new_model_prediction_performance.py
python script_old_model_hyperparameter_search.py
python script_old_model_precision_recall_f1.py
python script_old_model_prediction_performance.py
python script_significance_test.py
```


### Other Important Files

These files are needed for word embeddings using [GloVe](nlp.stanford.edu/data/wordvecs)

```
weights_matrix.npy          - contains the weight matrix that converts word to word embedding vectors
word_vector_mapping.json    - contains the mapping of words to the vectors in weights_matrix
```


## Getting Intermediate Results

Some of the scripts are dependent on other scripts. This is the order in which to run all the scripts and an explanation
of what the results of the intermediate steps are.

### 1. Getting the Predictions, Accuracy Scores and Confusion Matrices for the Baselines on the Test Set.

These scripts trains the n-gram models on the training set and outputs the predictions of those models together with the simple baselines to .csv files in a 'analyses' directory. Furthermore, it outputs the accuracy score tables and confusion matrices of all the baselines to .csv files in the 'analyses' directory.

```bash
python script_n_gram_models.py
python script_baselines.py
```


### 2 Getting the Predictions for the LSTM Models Given the Test Set.

These scripts trains the LSTM models on the training set and outputs the predictions of those models to .csv files in a 'analyses' directory.

```bash
python script_old_model_prediction_performance.py
python script_new_model_prediction_performance.py
```


### 3.1 Getting the Confusion Matrices for the LSTM Models Given the Test Set.

This script uses the predictions .csv of the previous step to calculate and store the confusion matrices of these models as .csv files in the 'analyses' directory.

```bash
python script_error_analysis.py
```


### 3.2 Getting the Accuracy Scores Tables for the LSTM Models Given the Test Set.

These scripts use the predictions .csv of the previous step to calculate and store the accuracy scores tables of these models as .csv files in the 'analyses' directory.

```bash
python script_old_model_precision_recall_f1.py
python script_new_model_precision_recall_f1.py
```


### 4 Getting the Graphs for the Accuracies of all the Models and the Error Analysis of all the models.

These scripts use the accuracy tables and confusion matrices created in the previous step and first step to visualise them as graphs and saves them in the 'analyses' directory.

```bash
python script_graphs_results.py
```


### 5 Significance Tests

These script performs significance tests between the model and the levels per model and outputs the results to a .csv file in the 'analyses' directory.

```bash
python script_significance_test.py
```


### 6 Getting the Graphs for the Statistics of the Data Set.

This script visualises the statistics of the data set as graphs and saves them in the 'analyses' directory.

```bash
python script_graphs_statistics.py
```


## Other Scripts

### The Hyperparameter Search

The following scripts can be editted for different search variables and perform a hyperparameter search on the LSTM models.

```bash
python script_old_model_hyperparameter_search.py
python script_new_model_hyperparameter_search.py
```