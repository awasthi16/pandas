🔹 1. Loading & Saving Data

pd.read_csv("file.csv") → Load CSV file.

pd.read_excel("file.xlsx") → Load Excel file.

df.to_csv("file.csv", index=False) → Save DataFrame to CSV.

df.to_excel("file.xlsx", index=False) → Save to Excel.

🔹 2. Exploring Data

df.head() → Show first 5 rows.

df.tail() → Show last 5 rows.

df.info() → Summary of columns & data types.

df.describe() → Statistical summary (mean, std, min, max, quartiles).

df.shape → Get rows & columns count.

df.columns → List of column names.

df.dtypes → Data types of each column.

df.isnull().sum() → Count missing values.

🔹 3. Selecting & Filtering

df['col'] → Select a single column.

df[['col1','col2']] → Select multiple columns.

df.iloc[0:5] → Select by index/position.

df.loc[0:5, ['col1','col2']] → Select by labels.

df[df['col'] > 10] → Conditional filtering.

df.query("col > 10 & col2 == 'A'") → SQL-like filtering.

🔹 4. Data Cleaning

df.dropna() → Remove missing rows.

df.fillna(value) → Fill missing values.

df.drop(columns=['col']) → Drop a column.

df.rename(columns={'old':'new'}) → Rename columns.

df.duplicated().sum() → Count duplicates.

df.drop_duplicates() → Remove duplicates.

🔹 5. Aggregation & Grouping

df['col'].value_counts() → Frequency of unique values.

df['col'].unique() → Unique values.

df['col'].nunique() → Count of unique values.

df.groupby('col').mean() → Group and aggregate.

df.pivot_table(values='val', index='col1', columns='col2', aggfunc='mean') → Pivot table.

🔹 6. Sorting

df.sort_values('col') → Sort by a column.

df.sort_values(['col1','col2'], ascending=[True, False]) → Multi-column sort.

🔹 7. Merging & Joining

pd.concat([df1, df2]) → Combine along rows/columns.

df1.merge(df2, on='col') → SQL-style join.

df1.join(df2, lsuffix='_1', rsuffix='_2') → Join on index.

🔹 8. Applying Functions

df['col'].apply(lambda x: x*2) → Apply custom function.

df.applymap(lambda x: str(x).upper()) → Apply function to entire DataFrame.

df['col'].map({'A':1,'B':2}) → Replace values using dictionary.

🔹 9. Reshaping Data

df.melt() → Convert wide → long format.

df.pivot(index='col1', columns='col2', values='col3') → Long → wide format.

df.stack() → Stack columns into rows.

df.unstack() → Unstack rows into columns.

🔹 10. Date & Time

pd.to_datetime(df['date']) → Convert to datetime.

df['date'].dt.year, .month, .day → Extract parts of date.

df.set_index('date') → Set datetime index.

df.resample('M').mean() → Resample time series (monthly avg).
