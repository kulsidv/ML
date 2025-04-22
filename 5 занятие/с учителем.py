import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def main():

    data = pd.read_csv("bikes_rent.csv")
    # тепловая карта корреляции между признаками
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    plt.show()

    # предобработка данных (удаление столбцов с высокой корреляцией)
    data = data.drop(["season", "atemp", "windspeed(ms)"], axis=1)
    # разделение на признаки и целевую переменную
    x, y = data.drop(["cnt"], axis=1), data["cnt"]

    # разделение на обучающую и тестовую выборки
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # обучение линейной регрессии и вывод коэффициента детерминации
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print(lr.score(x_test, y_test))

    predict = lr.predict(x_train)

    pca = PCA(n_components=2)
    pca.fit(x_train)
    x_pca = pca.transform(x_train)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_pca[:, 0], x_pca[:, 1], y_train, c=y_train, cmap='viridis')
    plt.show()


if __name__ == "__main__":
    main()
