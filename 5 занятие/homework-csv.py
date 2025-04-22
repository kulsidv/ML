import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import root_mean_squared_error


def main():

    data = pd.read_csv("AmesHousing.csv")
    # удаляем столбцы со строковым типом данных и заполняем нулями NaNы
    data = data.select_dtypes(exclude=['object']).fillna(0)
    # тепловая карта корреляции между признаками
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    plt.show()
    # посмотрела сколько пустых значений
    print(data.isnull().sum())

    # предобработка данных (удаление столбцов с высокой корреляцией)
    data = data.drop(["Order", "PID", "Total Bsmt SF", "TotRms AbvGrd",
                      "Garage Yr Blt", "Garage Cars", "Overall Qual"],
                     axis=1)
    # разделение на признаки и целевую переменную
    x, y = data.drop(["SalePrice"], axis=1), data["SalePrice"]

    # разделение на обучающую и тестовую выборки
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # обучение линейной регрессии и вывод коэффициента детерминации
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    predict = lr.predict(x_train)
    print(f'rmse = {root_mean_squared_error(y_test, predict, squared=False)}')

    # уменьшаем размерность данных до 2ух
    pca = PCA(n_components=2)
    pca.fit(x_train)
    x_pca = pca.transform(x_train)

    # рисуем 3d модель
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_pca[:, 0], x_pca[:, 1], y_train, c=y_train, cmap='viridis')
    plt.show()


if __name__ == "__main__":
    main()
