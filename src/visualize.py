import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


def visualize_tree(model, feature_names):
    plt.figure(figsize=(20, 10))

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["Did Not Survive", "Survived"],
        filled=True,
        rounded=True,
        fontsize=10
    )

    plt.title("Decision Tree - Titanic Survival Prediction")
    plt.show()