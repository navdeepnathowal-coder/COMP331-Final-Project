import pandas as pd

# Define column names from adult.names documentation
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

# Load training data
df_train = pd.read_csv("adult/adult.data", header=None, names=columns, na_values=" ?", skipinitialspace=True)

missing_counts = df_train.isnull().sum()
missing_pct = (df_train.isnull().sum() / len(df_train)) * 100

missing_summary = pd.DataFrame({
    "missing_count": missing_counts,
    "missing_pct": missing_pct.round(2)
}).sort_values("missing_pct", ascending=False)

print(missing_summary)

print(df_train.head())

# Load test data
df_test = pd.read_csv("adult/adult.test", header=0, names=columns, na_values=" ?", skipinitialspace=True)

# Remove trailing '.' in income labels
df_test["income"] = df_test["income"].str.replace(".", "", regex=False).str.strip()

print(df_test.head())

df_train.to_csv("adult/adult_train.csv", index=False)
df_test.to_csv("adult/adult_test.csv", index=False)


missing_summary.to_csv("results/missing_summary.csv")

invalid_age = df_train[(df_train["age"] < 0) | (df_train["age"] > 100)]
invalid_hours = df_train[(df_train["hours-per-week"] < 1) | (df_train["hours-per-week"] > 99)]

print("Invalid ages:", len(invalid_age))
print("Invalid hours:", len(invalid_hours))

# Save samples for evidence
invalid_age.head(20).to_csv("results/invalid_age_samples.csv", index=False)
invalid_hours.head(20).to_csv("results/invalid_hours_samples.csv", index=False)

valid_income = {"<=50K", ">50K"}
invalid_income = df_train[~df_train["income"].isin(valid_income)]

valid_sex = {"Male", "Female"}
invalid_sex = df_train[~df_train["sex"].isin(valid_sex)]

print("Invalid income labels:", len(invalid_income))
print("Invalid sex labels:", len(invalid_sex))

invalid_income.head(20).to_csv("results/invalid_income_samples.csv", index=False)
invalid_sex.head(20).to_csv("results/invalid_sex_samples.csv", index=False)


# Income distribution overall
print(df_train["income"].value_counts(normalize=True))

# Income distribution by sex
print(df_train.groupby("sex")["income"].value_counts(normalize=True))

# Income distribution by race
print(df_train.groupby("race")["income"].value_counts(normalize=True)) 

# bias and fairness analysis
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Make sure results folder exists
os.makedirs("results", exist_ok=True)

# Plot 1: Income distribution by sex
plt.figure(figsize=(6,4))
sns.countplot(data=df_train, x="sex", hue="income")
plt.title("Income Distribution by Sex")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("results/income_by_sex.png", dpi=200)
plt.close()

# Plot 2: Income distribution by race
plt.figure(figsize=(8,5))
sns.countplot(data=df_train, x="race", hue="income")
plt.title("Income Distribution by Race")
plt.xlabel("Race")
plt.ylabel("Count")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("results/income_by_race.png", dpi=200)
plt.close()

# Overall distribution
overall = df_train["income"].value_counts(normalize=True).round(3)

# By sex
by_sex = df_train.groupby("sex")["income"].value_counts(normalize=True).round(3)

# By race
by_race = df_train.groupby("race")["income"].value_counts(normalize=True).round(3)

# Save to CSV
overall.to_csv("results/income_distribution_overall.csv")
by_sex.to_csv("results/income_distribution_by_sex.csv")
by_race.to_csv("results/income_distribution_by_race.csv")