from utils.data_loader import data_load
from experiments.decision_tree import run as run_decision_tree
from experiments.random_forest import run as run_random_forest

import pandas as pd

X_train, X_test, y_train, y_test = data_load()

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# print("Random forest regressor:", run_decision_tree())
# print("Decision tree regressor:", run_random_forest())

if __name__ == "__main__":
    run_decision_tree()
    run_random_forest()
    
    
