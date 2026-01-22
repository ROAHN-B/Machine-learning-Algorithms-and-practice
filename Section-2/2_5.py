import pandas as pd
# Aggregation functions

# df.groupby("category_column")["numeric_column"].mean() # calculate mean of column
# df.groupby("category_column").agg({"numeric_column":["mean","max","min"]}) # calculate min, max and mean of the "category_column"


# pivot=df.pivot_table(
#     values = "numeric_column",
#     index="category_column",
#     aggfunc="mean"
# )


# def range_func(x):
#     return x.max()- x.min()

# df.groupby("category_column"["numeric_column"].agg(range_func))

# df.groupby("category_column"["numeric_column"].mean()) #to calculate mean, Aggregation includes min, mean and max
# df.groupby("category_column"["numeric_column"].max()) # to calculate max
# df.groupby("category_column"["numeric_column"].min()) # to calculate min


########EXERCISE###############

# data = {
#     "class": ["A", "B", "C", "A", "D"],
#     "score": [76, 45, 67, 90, 23],
#     "Age": [18, 28, 20, 22, 21],
# }

# df = pd.DataFrame(data)
# print("Original data: \n", df)

# # grouped = df.groupby("class").mean()
# # print(grouped)

# stats = df.groupby("class").agg(
#     {
#         "score": ["mean", "min", "max"],
#         "Age": ["mean", "min", "max"],
#     }  # Gives mean , min and max of "Score" and "Age"
# )

# print("This is the stats of the original table: \n", stats)
