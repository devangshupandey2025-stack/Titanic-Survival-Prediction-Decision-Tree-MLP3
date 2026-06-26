import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../data/train.csv")

plt.figure(figsize=(8,5))

plt.hist(
    df["age"].dropna(),
    bins=20
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.show()

df["survived"].value_counts().plot(
    kind="bar",
    figsize=(6,4)
)

plt.title("Survival Distribution")
plt.xlabel("Survived")
plt.ylabel("Number of Passengers")
plt.show()

plt.figure(figsize=(8,5))

plt.hist(
    df["fare"],
    bins=20
)

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")

plt.show()

print(pd.crosstab(df["sex"], df["survived"]))

pd.crosstab(df["sex"], df["survived"]).plot(
    kind="bar",
    figsize=(6,4)
)

plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.show()

print(pd.crosstab(df["pclass"], df["survived"]))

pd.crosstab(df["pclass"], df["survived"]).plot(
    kind="bar",
    figsize=(6,4)
)

plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.show()