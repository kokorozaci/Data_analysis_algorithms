from typing import TypeVar

import numpy as np
from sklearn import model_selection
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

X, y = load_iris(return_X_y=True)

# Для наглядности возьмем только первые два признака (всего в датасете их 4)
X = X[:, :2]


class KMeans:

    KMeans = TypeVar('KMeans', bound='KMeans')

    def __init__(self,
                 n_clusters: int = 5,
                 min_distance: float = 1e-4,
                 max_iter: int = 30,
                 random_state: int = None) -> None:
        self.n_clusters = n_clusters
        self.classes = np.arange(n_clusters)
        self.min_distance = min_distance
        self.max_iter = max_iter
        self.random_state = random_state
        self._centroids = None

    @property
    def centroids(self) -> np.array:
        return self._centroids

    def fit(self,
            X: np.array) -> KMeans:
        """
        Обучение модели, определяет центроиды кластеров
        """
        np.random.seed(self.random_state)
        self._centroids = X[:self.n_clusters].copy()
        for _ in range(self.max_iter):
            centroids_old = self._centroids.copy()
            labels = np.argmin(
                np.linalg.norm(
                    X - self._centroids[:, None],
                    axis=2),
                axis=0
            )
            for cl in self.classes:
                self._centroids[cl] = np.mean(X[labels == cl], axis=0)
            if np.max(np.linalg.norm(self._centroids - centroids_old, axis=1)) < self.min_distance:
                self._centroids = centroids_old
                break
        return self

    def predict(self,
                X: np.array) -> np.array:
        """
        Предсказывает классы для массива признаков X
        """
        return np.argmin(
            np.linalg.norm(
                X - self.centroids[:, None],
                axis=2),
            axis=0
        )

    def inner_distance(self,
                       X: np.array,
                       y: np.array) -> np.array:
        """
        Суммарное внутрикластерное расстояние
        """
        return np.sum(
            np.apply_along_axis(
                lambda x: np.sum(
                    np.linalg.norm(
                        X[y == x] - self.centroids[x],
                        axis=1)
                ), 1, self.classes[:, None]
            )
        )

    def visualize(self,
                  X: np.array,
                  y: np.array) -> None:
        """
        Визуализация кластеров
        """
        np.random.seed(52)  # при изменении параметра меняется цвет кластеров
        colors = np.random.sample((self.n_clusters, 3))

        plt.figure(figsize=(7, 7))

        # нанесем на график центроиды
        for centroid in self.centroids:
            plt.scatter(centroid[0], centroid[1], marker='x', s=130, c='black')

        # нанесем объекты раскрашенные по классам
        for i in range(X.shape[0]):
            plt.scatter(X[i, 0], X[i, 1], color=colors[y[i]])

        plt.title(f"Кластеризация KMeans, число кластеров n = {self.n_clusters}")
        plt.show()


for n in np.arange(1, 11):
    kmeans = KMeans(n_clusters=n,
                    min_distance=1e-4,
                    max_iter=30).fit(X)
    y_pred = kmeans.predict(X)
    print(kmeans.inner_distance(X, y_pred))
    kmeans.visualize(X, y_pred)
