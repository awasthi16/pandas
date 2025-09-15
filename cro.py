croco


# crocodile_analysis.py

import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Load Dataset
# =========================
df = pd.read_csv("global_crocodile_species.csv")

# -------------------------
# 📊 Basic Exploration
# -------------------------
print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Shape (rows, cols) ---")
print(df.shape)

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Summary Stats for Length & Weight ---")
print(df[['Observed Length (m)', 'Observed Weight (kg)']].describe())

# -------------------------
# 🔎 Filtering & Selection
# -------------------------
print("\n--- Length > 4m ---")
print(df[df['Observed Length (m)'] > 4])

print("\n--- Observed in India ---")
print(df[df['Country/Region'] == "India"])

print("\n--- Critically Endangered ---")
print(df[df['Conservation Status'] == "Critically Endangered"])

print("\n--- Selected Columns ---")
print(df[['Common Name', 'Scientific Name', 'Country/Region']])

print("\n--- Female Adults ---")
print(df[(df['Sex'] == "Female") & (df['Age Class'] == "Adult")])

# -------------------------
# 📂 Grouping & Aggregation
# -------------------------
print("\n--- Count per Country ---")
print(df['Country/Region'].value_counts())

print("\n--- Avg Length by Habitat ---")
print(df.groupby('Habitat Type')['Observed Length (m)'].mean())

print("\n--- Max Weight per Family ---")
print(df.groupby('Family')['Observed Weight (kg)'].max())

print("\n--- Species per Country ---")
print(df.groupby('Country/Region')['Scientific Name'].nunique())

print("\n--- Avg Length & Weight by Sex ---")
print(df.groupby('Sex')[['Observed Length (m)', 'Observed Weight (kg)']].mean())

# -------------------------
# 📌 Sorting
# -------------------------
print("\n--- Sorted by Length ---")
print(df.sort_values('Observed Length (m)', ascending=False))

print("\n--- Sorted by Country then Weight ---")
print(df.sort_values(['Country/Region', 'Observed Weight (kg)']))

print("\n--- Top 5 Heaviest ---")
print(df.nlargest(5, 'Observed Weight (kg)'))

# -------------------------
# 🧩 Advanced Filtering
# -------------------------
print("\n--- Long but Light (Length > 5m & Weight < 400kg) ---")
print(df[(df['Observed Length (m)'] > 5) & (df['Observed Weight (kg)'] < 400)])

print("\n--- Observer Name starts with 'Dr.' ---")
print(df[df['Observer Name'].str.startswith("Dr.", na=False)])

print("\n--- Notes containing 'injury' ---")
print(df[df['Notes'].str.contains("injury", case=False, na=False)])

print("\n--- Observations after 2020 ---")
print(df[pd.to_datetime(df['Date of Observation']) > "2020-01-01"])

print("\n--- Species in Multiple Countries ---")
print(df.groupby('Scientific Name')['Country/Region'].nunique().loc[lambda x: x > 1])

# -------------------------
# 📊 Statistics & Derived Columns
# -------------------------
print("\n--- Length Stats ---")
print(df['Observed Length (m)'].agg(['mean', 'median', 'std']))

def size_category(x):
    if x < 2:
        return "Small"
    elif x <= 4:
        return "Medium"
    else:
        return "Large"

df['Size Category'] = df['Observed Length (m)'].apply(size_category)
print("\n--- Size Category Counts ---")
print(df['Size Category'].value_counts())

print("\n--- Correlation Length vs Weight ---")
print(df[['Observed Length (m)', 'Observed Weight (kg)']].corr())

print("\n--- Age Class with Max Avg Weight ---")
print(df.groupby('Age Class')['Observed Weight (kg)'].mean().idxmax())

print("\n--- Avg Length per Conservation Status ---")
print(df.groupby('Conservation Status')['Observed Length (m)'].mean())

# -------------------------
# 📈 Visualization
# -------------------------
print("\n--- Saving Plots ---")

# Histogram of Length
plt.figure(figsize=(6,4))
df['Observed Length (m)'].hist(bins=20, color='skyblue', edgecolor='black')
plt.title("Histogram of Observed Length (m)")
plt.xlabel("Length (m)")
plt.ylabel("Count")
plt.savefig("hist_observed_length.png")
plt.close()

# Bar chart by Habitat
plt.figure(figsize=(6,4))
df['Habitat Type'].value_counts().plot(kind='bar', color='green', edgecolor='black')
plt.title("Observations per Habitat Type")
plt.xlabel("Habitat Type")
plt.ylabel("Count")
plt.savefig("bar_habitat_type.png")
plt.close()

print("✅ Plots saved as 'hist_observed_length.png' and 'bar_habitat_type.png'")
