# Text embedding fine tuning

## Before you start
### Set the virtual environment
```bash
$ make env
$ conda activate 01-openai-text-embedding-fine-tuning
$ make setup
```
### Dataset
- This dataset is a collection newsgroup documents.
- https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html
- https://www.kaggle.com/datasets/crawford/20-newsgroups

## Run the example
### Data preperation
```bash
$ export OPENAI_API_KEY=<YOUR API KEY>
$ python data_prep.py

# Data preperation for fine tuning
$ openai tools fine_tunes.prepare_data -f sport2.jsonl -q
$ ls data
sport2_prepared_train.jsonl sport2_prepared_valid.jsonl
```

### Fine tuning
```
# Fine tuning
$ openai api fine_tunes.create -t "data/sport2_prepared_train.jsonl" -v "data/sport2_prepared_valid.jsonl" --compute_classification_metrics --classification_positive_class " baseball" -m ada
Upload progress: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 1.52M/1.52M [00:00<00:00, 1.72Git/s]
...
...
Created fine-tune: <FINE TUNING KEY>

(Ctrl-C will interrupt the stream, but not cancel the fine-tune)
Streaming events until fine-tuning is complete...
[2023-05-14 15:20:35] Fine-tune costs $0.78
[2023-05-14 15:20:36] Fine-tune enqueued. Queue number: 0
[2023-05-14 15:20:37] Fine-tune started

# Get fine tuning results
$ openai api fine_tunes.results -i <FINE TUNING KEY> > result.csv
$ ls
result.csv
```
### Evaluation
- Plot the result.csv
```bash
$ python plot_eval.py
$ ls
accuracy.png
```

FYI, result.csv has the following columns.
```
step,elapsed_tokens,elapsed_examples,training_loss,training_sequence_accuracy,training_token_accuracy,validation_loss,validation_sequence_accuracy,validation_token_accuracy,classification/accuracy,classification/precision,classification/recall,classification/auroc,classification/auprc,classification/f1.0
```

### Get my fine-tune model
- https://platform.openai.com/docs/api-reference/fine-tunes/list
```bash
$ python get_fine_tune_model.py
$ ls 
fine_tune_model.json
```

### Inference
- required fine_tune_model.json
```
$ python inference.py
Fine tuning model: <YOUR MODEL NAME>
hockey
```

## Reference
- https://github.com/openai/openai-cookbook/blob/main/examples/Fine-tuned_classification.ipynb