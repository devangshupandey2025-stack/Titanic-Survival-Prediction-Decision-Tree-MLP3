from pathlib import Path
import pandas as pd

from src.preprocessing import preprocess_data
from src.train import train_decision_tree
from src.evaluate import evaluate_Decision_Tree
from src.visualize import visualize_tree

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR/"data"/"train.csv")
df = preprocess_data(df)

model, X_test, y_test, predictions = train_decision_tree(df)
evaluate_Decision_Tree(y_test, predictions)

visualize_tree(model, X_test.columns)