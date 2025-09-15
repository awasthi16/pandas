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


Load the dataset and display the first 5 rows.

Show the column names and their data types.

Find the total number of rows and columns.

Check if there are any missing values in the dataset.

Display summary statistics for numerical columns.



Retrieve all records where the crocodile’s average length is greater than 5 meters.

Select only Species, Continent, and Average_Length_m columns.

Find all crocodiles that live in Africa.

Get the species that are marked as "Critically Endangered".

Retrieve crocodiles whose population estimate is below 10,000.



Find the average weight of crocodiles grouped by continent.

Count how many species exist in each conservation status.

Find the maximum population estimate for each continent.

Get the mean and median length of crocodiles grouped by habitat type.

Which continent has the most crocodile species?



Sort crocodiles by average length in descending order.

Sort species by population estimate (lowest to highest).

Sort species first by continent, then by average weight.



Find crocodiles that are more than 4m long and weigh less than 400kg.

Get species whose diet contains the word "fish".

Select all species found in both Asia and Africa (if dataset allows multiple continents).

Find all crocodiles whose scientific name starts with "Crocodylus".



What is the overall mean length and weight of crocodiles worldwide?

Find the standard deviation of population estimates.

Identify the heaviest crocodile species per continent.

   
Replace missing values in Population_Estimate with the column’s mean.

Standardize text values in Continent column (e.g., "africa", "Africa" → "Africa").

Drop duplicate records (if any).

Visualization (with Pandas/Matplotlib)

Plot a bar chart showing the number of species per continent.

Plot a histogram of crocodile average lengths.



