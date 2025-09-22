import pandas as pd

data = {
    "PassengerId": [1, 2, 3, 4, 5, 6],
    "Name": ["John", "Sara", "Mike", "Emma", "Chris", "Sophia"],
    "Age": [22, 38, 26, 35, 28, None],
    "Sex": ["male", "female", "male", "female", "male", "female"],
    "Survived": [0, 1, 1, 1, 0, 1],
    "Fare": [7.25, 71.83, 8.05, 53.10, 8.46, 30.50]
}

df = pd.DataFrame(data)
print(df)
---------------------------------
df.head()      # first 5 rows
df.tail(3)     # last 3 rows
df.shape       # rows, columns
df.info()      # datatypes and null values
df.describe()  # statistics (mean, std, min, max, etc.)
------------
df.isnull().sum()         # count missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)  # fill missing with mean
------------------
df['Name']                 # select single column
df[['Name','Age']]         # select multiple columns
df[df['Age'] > 30]         # filter rows
df[(df['Sex']=="female") & (df['Survived']==1)]  # filter multiple conditions
------------------
df.sort_values(by="Age", ascending=True)
--------------
df.groupby("Sex")['Survived'].mean()
------------------
df['Sex_encoded'] = df['Sex'].map({"male":0, "female":1})
---------------
df.corr(numeric_only=True)
----------------
pd.pivot_table(df, values="Survived", index="Sex", aggfunc="mean")
--------------
df.to_csv("titanic_sample.csv", index=False)
-------------
df.rename(columns={"Sex":"Gender", "Survived":"Target"}, inplace=True)
df.drop("Fare", axis=1)          # drop column
df.drop([0,1], axis=0)           # drop rows by index
--------------
df.reset_index(drop=True, inplace=True)
--------------
df['Gender'].unique()       # unique values
df['Gender'].nunique()      # number of unique values
df['Gender'].value_counts() # frequency count
----------------------
df['AgeGroup'] = df['Age'].apply(lambda x: "Young" if x < 30 else "Old")
df['Gender'].replace({"male":"M", "female":"F"}, inplace=True)
df.sample(3, random_state=42)
df['Name'].str.upper()
df['Name'].str.contains("a")
df1 = pd.DataFrame({"PassengerId":[1,2,3], "Cabin":["C1","E1","B2"]})
merged = pd.merge(df, df1, on="PassengerId", how="left")
df2 = df.head(3)
df3 = df.tail(3)
pd.concat([df2, df3])
df.duplicated().sum()         # check duplicates
df.drop_duplicates(inplace=True)
df.query("Age > 30 and Gender == 'F'")
df['FareRank'] = df['Fare'].rank()
df['CumSumFare'] = df['Fare'].cumsum()
df['CumMeanFare'] = df['Fare'].expanding().mean()
df['RollingFare'] = df['Fare'].rolling(window=2).mean()
df['AgeBin'] = pd.cut(df['Age'], bins=[0,18,40,60], labels=["Teen","Adult","Senior"])
pd.melt(df, id_vars=["PassengerId"], value_vars=["Age","Fare"])
pd.crosstab(df['Gender'], df['Target'])
df = df.assign(NewFare=df['Fare']*2)
df.style.highlight_max(subset=["Age","Fare"], color="lightgreen")





-----------------------------------


📊 Basic Exploration

Load the dataset and display the first 5 rows.

Show the total number of observations (rows) and columns.

Display the data types of all columns.

Check if there are missing values in any column.

Display summary statistics for Observed Length (m) and Observed Weight (kg).

🔎 Filtering & Selection

Retrieve all observations where Observed Length (m) is greater than 4.

Find all crocodiles observed in India.

Get all entries where Conservation Status is "Critically Endangered".

Select only Common Name, Scientific Name, and Country/Region columns.

Find all observations where Sex is "Female" and Age Class is "Adult".

📂 Grouping & Aggregation

Count how many observations exist per Country/Region.

Find the average observed length grouped by Habitat Type.

Find the maximum observed weight for each Family.

Count the number of species (Scientific Name) observed per continent/country.

Find the average observed length and weight grouped by Sex.

📌 Sorting

Sort all observations by Observed Length (m) in descending order.

Sort observations by Country/Region and then by Observed Weight (kg).

Display the top 5 heaviest crocodile observations.

🧩 Advanced Filtering

Find crocodiles that are longer than 5m but weigh less than 400kg.

Get all entries where Observer Name starts with "Dr.".

Find observations where Notes column contains the word "injury".

Retrieve all observations made after "2020-01-01".

Find species (Scientific Name) that were observed in more than one country.

📊 Statistics & Derived Columns

Calculate the mean, median, and standard deviation of Observed Length (m).

Create a new column "Size Category":

Small (<2m)

Medium (2–4m)

Large (>4m)
Then count observations in each category.

Find the correlation between Observed Length (m) and Observed Weight (kg).

Which Age Class has the highest average observed weight?

For each Conservation Status, find the average observed length.

📈 Visualization

Plot a histogram of Observed Length (m).

Create a bar chart showing the number of observations per Habitat Type.






---------------------------------group by --------------------------
    Group by Sex and find the average Age of males and females.

ans: df.groupby("Sex")["Age"].mean()


Group by Survived and find the total number of passengers in each group.
    df.groupby("Survived")["PassengerId"].count()


Group by Sex and count how many survived in each category.
    df.groupby("Sex")["Survived"].sum()


Find the maximum Fare paid by survivors and non-survivors (Survived column).
df.groupby("Survived")["Fare"].max()



Group by Sex and calculate the average Fare paid.
    df.groupby("Sex")["Fare"].mean()


Group by Survived and calculate both the mean and median of Age.

Group by Sex and Survived, and count how many passengers are in each group.

Find the sum of Fare grouped by Survived.

Group by Sex and find the youngest and oldest passenger (Age).

Group by Survived and calculate the survival rate for males and females separately.
(Tip: You can use groupby(["Sex", "Survived"]).size() and then normalize)
