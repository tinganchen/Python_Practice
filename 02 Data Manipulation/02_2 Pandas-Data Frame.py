"""
Pandas, Python and Data Analysis / Panel Data

Outline

# 1. Pandas Data Structures
# (a) Series
# (b) Data frame
## Order of columns changed
## Some attribute's values used as index
## Transpose

# 2. I/O

# 3. Indexing
## df.set_index()
## df.sort_index(ascending = True)
# (a) Single attr. indexing
# (b) MultiIndex
## DF
## Tuple
## Array

# 4. Subsetting
# (a) Select specific columns
# (b) Select specific rows
# (c) Select specific columns and rows
# (d) Select specific cell

# 5. Filtering

# 6. Sorting

# 7. Update
# (a) Cell
# (b) Row
# (c) Column

# 8. Drop
# (a) Cell
# (b) Row
# (c) Column

# 9. Insert
# (a) Row
# (b) Column

# 10. Empty DF

# 11. Concatenate and Merge
# (a) concat()
# (b) merge()
# Attr.s with same names joined 
# Attr.s with different names joined
# Merge two tables on col and index
# Merge two tables with contradicted cols.

# 12. Group by

# 13. Pivot table

# 14. Apply
# operate for each column

# 15. Arithmetic

# 16. Missing value
# (a) isnull, notnull
# (b) dropna
# (c) fillna
"""

import pandas as pd

# 1. Pandas Data Structures
# (a) Series
s = pd.Series([3, -5, 7, 4], index=['a', 'b', 'c', 'd'])
s.index
s.values

attr = ['a', 'b', 'c', 'd']
attr_v_1 = [3, -5, 7, 4]
attr_v_2 = [0, 7, -7, 3]
s = pd.Series(attr_v_1, attr)
p = pd.Series(attr_v_2, attr)
(s + 2)*2
s + p
sum(s + p)
s["b"] # s[1]
s[["b", "d"]]

import numpy as np
s.apply(np.exp)
s.apply(np.abs)
np.exp(s)

# (b) Data frame
data = {'Country': ['Belgium', 'India', 'Brazil'], 
        'Capital': ['Brussels', 'New Delhi', 'Brasilia'],
        'Population': [11190846, 1303171035, 207847528]}
df = pd.DataFrame(data)
df2 = df
df2.index = [1, 2, 3]
# df2
# pd.DataFrame(data, index = [1, 2, 3])

## Order of columns changed
pd.DataFrame(data, index = [1, 2, 3],
             columns = ['Capital', 'Country', 'Population'])
df_col_ch = df
df_col_ch.columns = ['Capital', 'Country', 'Population']
# df_col_ch

## Some attribute's values used as index
df_ind = pd.DataFrame(data, columns = ['Capital', 'Population'],
                      index = data['Country'])

## Transpose
df_ind.T


# 2. I/O
df.to_html("02_2 Population.html")
df.to_csv("02_2 Population.csv", encoding="utf8")
# df.to_json
# df.to_excel

pd.read_html("02_2 Population.html")
df_csv = pd.read_csv("02_2 Population.csv")

df_csv.head(2)
df_csv.tail(1)

len(df_csv)
df_csv.shape
df_csv.info()

for index, row in df_csv.iterrows():
    print(index, row["Capital"], row["Population"])


# 3. Indexing
## df.set_index()
## df.sort_index(ascending = True)
    
# (a) Single attr. indexing
df_set_ind = pd.DataFrame(data).set_index('Country')
df_set_ind.sort_index(ascending = True)
df_set_ind.reset_index() # return index to attr. names

# (b) MultiIndex
market = {"City" : ["Taipei", "Tainan", "Hualien", "Taipei", "Taipei", "Tainan"], 
          "Product" : ["Apple", "Banana", "Orange", "Banana", "Grape", "Durian"],
          "Price" : [30, 10, 15, 25, 40, 100]}
df_mkt = pd.DataFrame(market, 
                      index = ["1st", "2nd", "3rd", "4th", "5th", "6th"])

## DF
#
ind_city_prod = df_mkt.set_index(['City', 'Product'])
sort_ind = ind_city_prod.sort_index(ascending = True, inplace = False)
    # inplace=True : replace current df
# print(sort_ind)
sort_ind[: "Orange"] 
sort_ind["Taipei": ] 

sort_ind.unstack()
sort_ind.stack()

#
df = pd.DataFrame(np.random.rand(4, 2),
                  index = [('Taipei', 10), ('Taipei', 20), ('Tainan', 10), ('Tainan', 20)],
                  columns = ['data1', 'data2'])
# df
index1 = pd.MultiIndex.from_product([['Taipei', 'Tainan'], [10, 20]])
df.reindex(index1)

## Tuple
index2 = pd.MultiIndex.from_tuples([('Taipei', 10), ('Taipei', 20), ('Tainan', 10), ('Tainan', 20)])
df.reindex(index2)

market1_2 = {("Taipei", "Apple"): 30, ("Tainan", "Banana"): 10,
             ("Hualien", "Orange"): 15, ("Taipei", "Banana"): 25, 
             ("Taipei", "Grape"): 40, ("Tainan", "Durian"): 100}
pd.Series(market1_2)
pd.Series(market1_2).sort_index(ascending = True, inplace = False)

## Array
index3 = pd.MultiIndex.from_arrays([['Taipei', 'Taipei', 'Tainan', 'Tainan'], [10, 20, 10, 20]])
reindex3 = df.reindex(index3)

reindex3.index.names = ["City", "Number"]
# reindex3

reindex3.unstack(level = 0) # level = 1

reindex3.mean(level = "City")


# 4. Subsetting
# (a) Select specific columns
df_mkt["City"]
df_mkt[["City", "Product"]]

df_mkt.iloc[:, [0, 2]]

# (b) Select specific rows
df_mkt.head(3)
df_mkt.tail(2)
df_mkt[0:2]

df_mkt.loc["1st"] # Error: df_mkt["1st"]

df_mkt.iloc[0]
df_mkt.iloc[0:2]
df_mkt.iloc[0:2, :]

# (c) Select specific columns and rows
df_mkt.loc["1st", "Price"]
df_mkt.loc["1st", ["City", "Product"]] # select 2nd row 
df_mkt.loc[:, ["City", "Product"]] # :, select all rows
df_mkt.loc["3rd" : "5th", ["City", "Product"]]

df_mkt.iloc[0:2, 1:3]
df_mkt.iloc[[1, 2, 4], [0, 2]]

# (d) Select specific cell
df_mkt.iat[1, 1]


# 5. Filtering
df_mkt[df_mkt["Price"] > 10]
df_mkt[(df_mkt["Price"] > 10) & (df_mkt["Price"] < 30)]

df_mkt[df_mkt["City"].isin(["Taipei", "Tainan"])]


# 6. Sorting
"""Recall # 3.(b)---------
ind_city_prod = df_mkt.set_index(['City', 'Product'])
sort_ind = ind_city_prod.sort_index(ascending = True, inplace = False)

"""
sort_ind.sort_index(ascending = True).sort_values(['City', 'Product', 'Price'], ascending = False)

df_mkt.sort_values(['City', 'Product', 'Price'], ascending = False)


# 7. Update
# (a) Cell
update_df_mkt = df_mkt.copy()
update_df_mkt.loc["1st", "Price"] = 35
# update_df_mkt
update_df_mkt.iloc[0, 2] = 40

# (b) Row
update_df_mkt.loc["2nd"] = ["Taichung", "Banana", 5]

# (c) Column
"""import numpy as np"""
np.random.seed(123)
update_df_mkt.loc[:, "Price"] = np.random.randint(5, 50, size = len(update_df_mkt))


# 8. Drop
# (a) Cell
drop_df_mkt = df_mkt.copy()
drop_df_mkt.loc["1st", "Price"] = None
# drop_df_mkt
drop_df_mkt.iloc[1, 2] = None

# (b) Row
drop_df_mkt.drop(["3rd", "4th"])

drop_df_mkt.drop(drop_df_mkt.index[[2, 3]])

# (c) Column
drop_df_mkt.drop(["City"], axis = 1)


# 9. Insert
# (a) Row
insert_df_mkt = df_mkt.copy()
insert_df_mkt.loc["7th"] = ["Taipei", "Guava", 13]

s = pd.Series({"City":"Taipei", "Product":"Papaya", "Price":20})
insert_df_mkt.append(s, ignore_index = True)

# (b) Column
insert_df_mkt["Quantity"] = pd.Series([np.random.randint(5, 10) 
                                        for i in range(len(insert_df_mkt))]).values
insert_df_mkt.loc[:, "Q2"] = np.random.randint(5, 10, len(insert_df_mkt))


# 10. Empty DF
empty_df = pd.DataFrame(np.nan, index = df_mkt.index.values, columns = df_mkt.columns)


# 11. Concatenate and Merge
# (a) concat()
market2 = {"Price":[30, 10], "Product":["Cherry", "Orange"]}
df_mkt2 = pd.DataFrame(market2)
pd.concat([df_mkt, df_mkt2], ignore_index = True) # , ignore_index = False
pd.concat([df_mkt, df_mkt2], keys = ["mkt1", "mkt2"]) # Keys work when ignore_index = False

pd.concat([df_mkt, df_mkt2], join = "outer") # Union(col)
pd.concat([df_mkt, df_mkt2], join = "inner") # Intersect(col)

# (b) merge()
# Attr.s with same names joined 
market3 = {"Quatity":[30, 10], "Product":["Cherry", "Orange"]}
df_mkt3 = pd.DataFrame(market3)
pd.merge(df_mkt, df_mkt3)
pd.merge(df_mkt, df_mkt3, on = "Product")
pd.merge(df_mkt, df_mkt3, how = "outer") # Union(row)
pd.merge(df_mkt, df_mkt3, how = "inner") # Intersect(row)
pd.merge(df_mkt, df_mkt3, how = "left")
pd.merge(df_mkt, df_mkt3, how = "right")

# Attr.s with different names joined
market4 = {"Quatity":[30, 10], "Fruit":["Cherry", "Orange"]}
df_mkt4 = pd.DataFrame(market4)
pd.merge(df_mkt, df_mkt4, 
         left_on = "Product", right_on = "Fruit") 
pd.merge(df_mkt, df_mkt4, 
         left_on = "Product", right_on = "Fruit").drop("Fruit", axis = 1) 

# Merge two tables on col and index
pd.merge(df_mkt, df_mkt4.set_index("Fruit"), 
         left_on = "Product", right_index = True)

# Merge two tables with contradicted cols.
mkt2 = {"City" : ["Taipei", "Tainan", "Hualien", "Taipei", "Taipei", "Tainan"], 
        "Product" : ["Cucumber", "Carrot", "Spinach", "Carbage", "Tomato", "Potato"],
        "Price" : [15, 10, 30, 40, 25, 100]}
df_mkt_2 = pd.DataFrame(mkt2)
pd.merge(df_mkt, df_mkt_2, on = "City")
pd.merge(df_mkt, df_mkt_2, on = "City", suffixes = ["_Fruit", "_Vegetable"])


# 12. Group by
df_mkt.groupby("City").sum()
df_mkt.groupby("City").mean()
df_mkt.groupby(["City", "Product"]).mean()


# 13. Pivot table
df_mkt.pivot_table(index = "City", columns = "Product", values = "Price")


# 14. Apply
# operate for each column
df_mkt["Quantity"] = [np.random.randint(3, 10) for i in range(len(df_mkt))]
df_mkt_pq = df_mkt.iloc[:, [2, 3]]

df_mkt_pq.apply(np.max)
df_mkt_pq.apply(lambda x: x.max()-x.min())
def square(alist):
    for i in range(len(alist)):
        return alist**2
df_mkt_pq.apply(square)


# 15. Arithmetic
A = pd.DataFrame([[1, 11], [5, 1]])
A.mean() # applied by Column
A.stack().mean()

B = pd.DataFrame(np.random.randint(0, 10, (3, 3)))

A+B # broadcasting
A.add(B)
A.add(B, fill_value = 0)


# 16. Missing value

# (a) isnull, notnull
A = pd.Series([1, np.nan, None])
A.isnull()
A[A.notnull()]

DF = pd.DataFrame([[1, 11, np.nan],
                  [2, 3, 8],
                  [np.nan, 1, 0]])
DF.isnull()

# (b) dropna
A.dropna()

DF.dropna() # Drop rows with NA
DF.dropna(axis = 1) # Drop columns with NA
# DF.dropna(axis = "columns")

DF2 = DF.copy()
DF2[2] = np.nan
DF2.dropna(axis = 1, how = "all") # Drop columns with all NAs
DF2.dropna(axis = 1, thresh = 2) # Drop columns with 2 or more NAs

# (c) fillna
A.fillna(0)
fill = A[A.notna()].mean()
A.fillna(fill)

DF.fillna(0)
DF.fillna(DF.mean(0)) # Fill with mean for each column
# DF.fillna?
