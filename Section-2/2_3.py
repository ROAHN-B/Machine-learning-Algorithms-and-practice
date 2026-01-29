# import pandas as pd

# s = pd.Series(
#     [10, 20, 30, 40], index=["a", "b", "c", "d"]
# )  # creates key and value pair
# print(s)

# s["e"] = 50
# print(s)
# data = {"names": ["Rohan", "Belsare"], "Age": [24, 45]}
# df = pd.DataFrame(data)  # Creates two dimentional labelled data
# print(df)

#### DATA LOADING FROM CSV, EXEL OR ETC##############

# loading data from CSV and excel
# df = pd.read_csv("data.csv")
# df = pd.read_excel("data.xlsx")

# df.to_csv("data.csv") # to save a csv file

####### BASIC DATA FRAME OPERATION##############
# print(df.head())  # prints 1st 5 elements of dataframe
# print(df.tail())  # prints last 5 elements of dataframe

# print(df.info())  # gives summary of datagrame
# print(df.describe())  # gives statistical summary of dataframe


# print(df["name"])   # prints columns of name and age

# print(df[df["age"] > 25])  # prints age from age table where age is greater than 25

# print(df.iloc[0])  # prints 1st row by position
# print(df.iloc[:, 0]) # prints 1st column by position


########EXERCISE##########

# df = pd.read_csv("Iris.csv")
# selected_columns = df[["Species", "PetalWidthCm"]]
# print("selected columns: \n", selected_columns)


# filtered_rows = df[(df["SepalLengthCm"] > 5.0) & (df["Species"] == "Iris-setosa")]
# print("Filtered columns: \n", filtered_rows)

# filtered_rows.to_csv("data1.csv")
