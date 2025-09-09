Creating Data

pd.Series()

pd.DataFrame()

pd.read_csv(), pd.to_csv()

pd.read_excel(), pd.to_excel()

pd.read_json(), pd.to_json()

pd.read_sql(), pd.to_sql()

pd.read_html()

pd.read_clipboard()

pd.DataFrame.from_dict(), pd.DataFrame.from_records()

2. Basic Inspection

df.head(), df.tail()

df.info()

df.shape

df.columns

df.index

df.dtypes

df.describe()

df.memory_usage()

3. Selection & Indexing

df['col'], df[['col1','col2']]

df.loc[] → label-based

df.iloc[] → position-based

df.at[], df.iat[]

df.filter()

df.query()

4. Sorting & Ranking

df.sort_values(by=...)

df.sort_index()

df.rank()

5. Missing Data

df.isna(), df.notna()

df.fillna(value)

df.dropna()

df.interpolate()

6. Statistics / Math

df.sum(), df.mean(), df.median(), df.mode()

df.min(), df.max()

df.std(), df.var()

df.corr(), df.cov()

df.cumsum(), df.cumprod()

7. String Operations (Series.str)

series.str.lower(), series.str.upper()

series.str.contains()

series.str.replace()

series.str.split(), series.str.cat()

series.str.strip()

8. Datetime Operations (Series.dt)

series.dt.year, series.dt.month, series.dt.day

series.dt.hour, series.dt.minute

series.dt.weekday

series.dt.strftime()

9. GroupBy & Aggregation

df.groupby('col')

group.agg(['mean','sum','count'])

group.transform()

group.size()

10. Merging & Joining

pd.concat([df1,df2])

pd.merge(df1,df2,on=...,how='inner/outer/left/right')

df.join(other_df)

11. Reshaping

df.pivot()

df.pivot_table()

df.melt()

df.stack(), df.unstack()

df.explode()

12. Window Functions

df.rolling(window=3).mean()

df.expanding().sum()

df.ewm(span=3).mean()

13. Input/Output (I/O)

pd.read_csv(), df.to_csv()

pd.read_excel(), df.to_excel()

pd.read_parquet(), df.to_parquet()

pd.read_hdf(), df.to_hdf()

pd.read_sql(), df.to_sql()

14. Utility Functions

pd.concat()

pd.merge()

pd.get_dummies()

pd.cut(), pd.qcut()

pd.unique(), pd.value_counts()

pd.crosstab()

pd.pivot_table()
