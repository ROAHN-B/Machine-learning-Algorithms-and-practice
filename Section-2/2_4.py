# HANDLING MISSING VALUES
import pandas as pd
import numpy as np

# df1 = pd.read_csv("Iris.csv")
# df2 = pd.read_csv("Iris.csv")
# df = df.dropna()
# df = df.dropna(axis=1)  # drops columns with missing values

# df["column_name"] = df["column_name"].fillna(
#     0
# )  # Fills the missing places in colum with "0"
# df.fillna(
#     method="ffill"
# )  # Fills missing values according to the data it is forward fill
# df.fillna(
#     method="bfill"
# )  # Fills missing values according to the data it is backward fill

# # INTERPOLATION
# df["column_name"] = df["column_name"].interpolate()

# DATA TRANSFORMATION ex- Renaming columns, Data Types

# df = df.rename(columns={"Species": "Species_name"})  # changing column name
# df["SepalLengthCm"] = df["SepalLengthCm"].astype("int") # changing datatype of a specific column in dataset

# COMBINING AND MERGING DATAFRAMES

# combined_rows = pd.concat(
#     [df1, df2], axis=0
# )  # combines dataframes of two dataset along rows
# combined_columns = pd.concat(
#     [df1, df2], axis=1
# )  # combine dataframes of two datasets along columns

# print("combined_rows: \n", combined_rows)
# print("combined_columns: \n", combined_columns)

# combined_columns.to_csv("data2.csv")
# combined_rows.to_csv("data2.csv")


# merged = pd.merge(df1, df2, on="Species")  # Merging data based on common column name
# merged = pd.merge(df1, df2, how="left", on="Species")  # Performns left join
# merged = pd.merge(df1, df2, how="inner", on="Species")  # Performns inner join

# joined = df1.join(df2, how="inner") #joining dataset based on indexes


#######EXERCISE######################
# data1 = {"name": ["Rohan", "Laukik", "Abhijeet", np.nan], "Age": [20, 20, 23, np.nan]}
# data2 = {"name": ["Rohan", "Laukik", "Abhijeet", np.nan], "Age": [20, 20, 23, np.nan]}
# df1 = pd.DataFrame(data1)
# df2 = pd.DataFrame(data2)
# print("original data1: \n", df1)
# print("original data1: \n", df1)

# merged = pd.merge(df1, df2, how="inner", on="name")
# print("Merged Data: \n", merged)
