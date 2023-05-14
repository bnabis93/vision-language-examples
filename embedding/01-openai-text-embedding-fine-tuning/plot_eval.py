import pandas as pd

results = pd.read_csv("result.csv")
results[results["classification/accuracy"].notnull()][
    "classification/accuracy"
].plot().get_figure().savefig("accuracy.png")
