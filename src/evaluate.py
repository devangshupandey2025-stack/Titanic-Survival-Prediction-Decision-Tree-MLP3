from sklearn.metrics import(
    accuracy_score,
    confusion_matrix,
    classification_report
)

def evaluate_Decision_Tree(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy : {accuracy:.4f}")
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, predictions))
    print("Classification Report: ")
    print(classification_report(y_test, predictions))
    