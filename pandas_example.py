import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    "Name": ["Amit", "Ravi", "Priya", "Neha", "Rohit"],
    "Age": [25, 30, 22, 28, 35],
    "Salary": [50000, 60000, 55000, 65000, 70000],
    "Department": ["IT", "HR", "IT", "Finance", "HR"]
}

df = pd.DataFrame(data)
print(df)


1. Basic Inspection

df.head(3)       # first 3 rows
df.tail(2)       # last 2 rows
df.info()        # summary (columns, types, nulls)
df.shape         # (rows, cols)
df.describe()    # statistics (mean, std, min, max)

2. Selection & Indexing
df['Name']             # single column
df[['Name','Salary']]  # multiple columns
df.iloc[0:2]           # first 2 rows
df.loc[df['Age'] > 25] # filter condition

3. Sorting
df.sort_values(by="Age")           # ascending
df.sort_values(by="Salary", ascending=False)  # descending


4. Missing Data
df2 = df.copy()
df2.loc[2, "Salary"] = np.nan

df2.isna()                 # check NaN
df2.fillna(0)              # replace NaN with 0
df2.dropna()               # drop rows with NaN

5. Math / Statistics
df['Salary'].mean()   # average
df['Salary'].sum()    # total
df['Age'].min()       # minimum
df['Age'].max()       # maximum
df['Age'].std()       # standard deviation

6. String Operations
df['Name'].str.upper()       # uppercase
df['Department'].str.contains("IT")  # filter IT dept
df['Name'].str.replace("a", "@")     # replace chars

7. Datetime
dates = pd.to_datetime(["2023-01-01", "2023-03-15", "2023-06-20"])
df["Joining_Date"] = dates

df["Joining_Date"].dt.year     # extract year
df["Joining_Date"].dt.month    # extract month

8. GroupBy & Aggregation
df.groupby("Department")["Salary"].mean()
# Average salary by department

df.groupby("Department").agg({
    "Age": "mean",
    "Salary": ["min", "max"]
})

9. Merging / Joining
dept_info = pd.DataFrame({
    "Department": ["IT", "HR", "Finance"],
    "Manager": ["Sohan", "Meera", "Vikram"]
})

pd.merge(df, dept_info, on="Department", how="left")

10. Reshaping
# Pivot table
df.pivot_table(values="Salary", index="Department", aggfunc="mean")

# Melt (unpivot)
pd.melt(df, id_vars=["Name"], value_vars=["Age","Salary"])

11. Value Counts & Dummies
df["Department"].value_counts()    # count categories
pd.get_dummies(df["Department"])   # one-hot encoding

12. Rolling / Window
df["Salary"].rolling(window=2).mean()
# Moving average of salary (2 rows window)

