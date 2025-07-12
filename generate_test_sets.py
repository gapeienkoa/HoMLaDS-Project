import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./data/topic21_v3_train.csv")

target = "price"

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print(f"Размер обучающей выборки: {X_train.shape[0]}")
print(f"Размер тестовой выборки: {X_test.shape[0]}")

X_train.to_csv("./data/X_train.csv", index=False)
X_test.to_csv("./data/X_test.csv", index=False)
y_train.to_csv("./data/y_train.csv", index=False)
y_test.to_csv("./data/y_test.csv", index=False)

print("Данные успешно сохранены в папку /data.")