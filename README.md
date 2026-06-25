# Problem Statement

The sinking of the RMS Titanic is one of the most well-known maritime disasters in history. During the tragedy, not all passengers had the same chance of survival. Factors such as passenger class, age, gender, fare, and family size played a significant role in determining whether a passenger survived.

The objective of this project is to build a **Decision Tree Classification** model that predicts whether a passenger survived the Titanic disaster based on their demographic and travel information. Using the Titanic dataset, the project demonstrates the complete machine learning workflow, including data loading, preprocessing, feature selection, model training, evaluation, and visualization.

The primary goal is to classify each passenger into one of two categories:

* **0** → Did Not Survive
* **1** → Survived

This project serves as a practical implementation of the Decision Tree algorithm and provides hands-on experience with real-world data preprocessing, handling missing values, encoding categorical features, model evaluation, and interpreting decision tree visualizations.

# Dataset

This project uses the **Titanic Survival Dataset**, one of the most popular datasets for learning and practicing Machine Learning classification algorithms.

Each row in the dataset represents a passenger aboard the RMS Titanic, while each column contains information about that passenger. The goal is to predict whether a passenger survived the disaster.

## Dataset Information

* **Dataset Name:** Titanic Survival Dataset
* **Problem Type:** Binary Classification
* **Target Variable:** `survived`

| Value | Meaning                   |
| ----- | ------------------------- |
| 0     | Passenger Did Not Survive |
| 1     | Passenger Survived        |

## Features

| Feature    | Description                                               |
| ---------- | --------------------------------------------------------- |
| `pclass`   | Passenger ticket class (1 = First, 2 = Second, 3 = Third) |
| `name`     | Passenger's full name                                     |
| `sex`      | Gender of the passenger                                   |
| `age`      | Age of the passenger (in years)                           |
| `sibsp`    | Number of siblings/spouses aboard the Titanic             |
| `parch`    | Number of parents/children aboard the Titanic             |
| `ticket`   | Ticket number                                             |
| `fare`     | Passenger fare paid                                       |
| `cabin`    | Cabin number assigned to the passenger                    |
| `embarked` | Port where the passenger boarded the ship                 |
| `survived` | Target variable indicating survival (0 = No, 1 = Yes)     |

## Dataset Characteristics

* **Total Records:** 891
* **Total Features:** 10 Input Features + 1 Target Feature
* **Target Classes:** 2 (Survived / Did Not Survive)
* **Learning Type:** Supervised Learning
* **Algorithm Used:** Decision Tree Classifier

## Challenges in the Dataset

The dataset contains several real-world challenges that require preprocessing before training the model:

* Missing values in features such as **Age**, **Cabin**, and **Embarked**.
* Categorical features (e.g., **Sex**, **Embarked**) that must be encoded into numerical values.
* Irrelevant features (e.g., **Name**, **Ticket**, **Cabin**) that may not contribute significantly to prediction.
* A mix of numerical and categorical data, making it an excellent dataset for learning data preprocessing techniques.
