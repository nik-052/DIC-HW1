import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# reading data
df = pd.read_csv("data/linear_regression.csv")

print(df.head(10))

print(df.info())

print(df.describe())

print(list(df.select_dtypes(include=["float64"]).columns))
print("Length of select columns: ", len(df.select_dtypes(include=["int64","float64"]).columns))

# check for Null values
for col in df.columns:
    if df[col].notna().sum() < 2000 :
        print(col)

print(df.isna().sum())


# check for skewness
# for col in df.columns:
#     plt.hist(df[col])
#     plt.xlabel(col)
#     plt.show()


X = df.drop(columns="Target")
y = df["Target"]

model = LinearRegression()

model.fit(X,y)

y_predict = model.predict(X)

diff = y-y_predict

print("Residual Description",diff.describe())

# Since the minimum residual error is -21 , the maximum being 3.44 and the mean error being 0.18 
# there are few samples that have huge deviation , lets filter them

t = 3 * diff.std()

# plt.figure(figsize=(10, 6))
# sns.histplot(diff, bins=100, kde=True, color='cyan')
# plt.axvline(-t, color='red', linestyle='-', label=f'Threshold (-3σ)')
# plt.title('Residuals Distribution with Outliers Threshold')
# plt.xlabel('Residuals')
# plt.ylabel('Frequency')
# plt.legend()
# plt.show()


# get the index of the mislabbeled data
mismatchedData = diff[diff < -t].index

print("Total number of mismatch samples",len(mismatchedData))
print("Mislabelled index: ",mismatchedData)
print(df.loc[mismatchedData])


# creating a column to define if the sample is mislabeled
df['Outlier'] = df.index.isin(mismatchedData).astype(int)
df.to_csv("linear_regression_done.csv")

