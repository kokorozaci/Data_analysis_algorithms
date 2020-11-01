#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from typing import TypeVar, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import gc # сборщик мусора
np.random.seed(4)


# In[5]:


df = pd.read_csv('H:/PyProgects/Tutors_expected_math_exam_results/train.csv')


# In[6]:


df.head()


# In[7]:


x = df.drop('mean_exam_points', axis=1)
y = df[['mean_exam_points']]

x_final = pd.read_csv('H:/PyProgects/Tutors_expected_math_exam_results/test.csv')

# сразу создам
preds_final = pd.DataFrame()
preds_final['Id'] = x_final['Id'].copy()

x.set_index('Id', inplace=True)
x_final.set_index('Id', inplace=True)

print('Строк в трейне:' ,  x.shape[0])
print('Строк в тесте', x_final.shape[0])

# Удалим ненужные файлы
del df
gc.collect()  


# In[8]:


x.describe()


# In[9]:


x.head()


# In[10]:


corr = x[['age', 'years_of_experience', 'lesson_price', 'qualification']].corr()
plt.figure(figsize = (16, 8))
mask = np.zeros_like(corr, dtype=np.bool)  # отрезаем лишнюю половину матрицы
mask[np.triu_indices_from(mask)] = True
sns.set(font_scale=1.4)
sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', linewidths=.5, cmap='GnBu')
plt.title('Correlation matrix')
plt.show();


# In[11]:


# Box plot
plt.figure(figsize=(12, 6))
sns.boxplot(x['years_of_experience'], y['mean_exam_points'])
plt.xlabel('qualification')
plt.ylabel('mean_exam_points')
plt.title('Distribution of Price by Rooms')
plt.show();


# In[12]:


x.groupby(['qualification'], as_index=False).agg({'lesson_price':'median'})


# In[13]:


x.columns


# In[14]:


# выбор дополнения для площади
plt.scatter(x['lesson_price'], y['mean_exam_points'])

plt.show()


# In[15]:


# Box plot
plt.figure(figsize=(12, 6))
sns.boxplot(x['history'], y['mean_exam_points'])
plt.xlabel('qualification')
plt.ylabel('mean_exam_points')
plt.title('Distribution of Price by Rooms')
plt.show();


# In[16]:


x['history'].value_counts()


# In[17]:


y.loc[x.index.isin(x.loc[x['history'] == 1].index)]


# In[18]:


class FeatureGenetator():
    """Генерация новых фич"""
    
    def __init__(self):
        self.med_exam_points_by_qualification = None
        self.med_lesson_price_by_qualification = None
        
    def fit(self, X, y=None):
        
        df = X.copy()
        
        self.med_lesson_price_by_qualification = x.groupby(['qualification'], as_index=False).agg({'lesson_price':'median'}).                                            rename(columns={'lesson_price':'MedPriceByQualification'})
        
        if y is not None:
            df['mean_exam_points'] = y.values
            self.med_exam_points_by_qualification = df.groupby(['qualification'], as_index=False).agg({'mean_exam_points':'median'}).                                            rename(columns={'mean_exam_points':'MedPointsByQualification'})
            
    def transform(self, X):
        
        if self.med_lesson_price_by_qualification is not None:
            X = X.merge(self.med_lesson_price_by_qualification, on=['qualification'], how='left')
            X['MedPriceByQualification'] = X['lesson_price'] - X['MedPriceByQualification']
        
        if self.med_exam_points_by_qualification is not None:
            X = X.merge(self.med_exam_points_by_qualification, on=['qualification'], how='left')
            
        return X


# In[19]:


features = FeatureGenetator()

features.fit(x, y)

x = features.transform(x)
x_final =  features.transform(x_final)


# In[20]:


x_final.head()


# In[25]:



from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(x.values, 
                                                                                     y.values, 
                                                                                     test_size = 0.3,
                                                                                     random_state = 1)


# In[22]:


class Node:
    Node = TypeVar('Node', bound='Node')

    def __init__(self,
                 index: int,
                 t: float,
                 true_branch: Node,
                 false_branch: Node) -> None:
        self.index = index  # индекс признака, по которому ведется сравнение с порогом в этом узле
        self.t = t  # значение порога
        self.true_branch = true_branch  # поддерево, удовлетворяющее условию в узле
        self.false_branch = false_branch  # поддерево, не удовлетворяющее условию в узле


class RegressorLeaf:

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.prediction = self.predict()

    def predict(self):
        prediction = np.mean(self.labels)
        return prediction


class TreeBuilder:
    _leaf = None

    def __init__(self,
                 max_depth: int = None,
                 min_samples_leaf: int = 1,
                 min_samples_split: int = 2) -> None:
        self.max_depth = max_depth
        self.min_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self._tree = None
        self.tree_depth = 0
        self.criterion = None

    def quality(self,
                left_y: np.array,
                right_y: np.array,
                current_criteria: float) -> float:
        p = float(left_y.shape[0]) / (left_y.shape[0] + right_y.shape[0])
        return current_criteria - p * self.criterion(left_y) - (1 - p) * self.criterion(right_y)

    @staticmethod
    def split(X: np.array,
              y: np.array,
              index: int,
              t: int) -> Tuple[np.array, np.array, np.array, np.array]:

        left = np.where(X[:, index] <= t)
        right = np.where(X[:, index] > t)

        true_data, false_data = X[left], X[right]
        true_labels, false_labels = y[left], y[right]

        return true_data, false_data, true_labels, false_labels

    def find_best_split(self, data, labels):

        current_criteria = self.criterion(labels)

        best_quality = 0
        best_t = None
        best_index = None

        n_features = data.shape[1]

        for index in range(n_features):
            # будем проверять только уникальные значения признака, исключая повторения
            t_values = np.unique([row[index] for row in data])

            for t in t_values:
                true_data, false_data, true_labels, false_labels = self.split(data, labels, index, t)
                #  пропускаем разбиения, в которых в узле остается менее 5 объектов
                if len(true_data) < self.min_leaf or len(false_data) < self.min_leaf:
                    continue

                current_quality = self.quality(true_labels, false_labels, current_criteria)

                #  выбираем порог, на котором получается максимальный прирост качества
                if current_quality > best_quality:
                    best_quality, best_t, best_index = current_quality, t, index

        return best_quality, best_t, best_index

    def build_tree(self, data, labels):

        quality, t, index = self.find_best_split(data, labels)

        #  Базовый случай - прекращаем рекурсию, когда нет прироста в качества
        if quality == 0:
            return self._leaf(data, labels)

        if self.max_depth and self.tree_depth >= self.max_depth:
            return self._leaf(data, labels)

        if len(data) <= self.min_samples_split:
            return self._leaf(data, labels)

        self.tree_depth += 1

        true_data, false_data, true_labels, false_labels = self.split(data, labels, index, t)

        # Рекурсивно строим два поддерева
        true_branch = self.build_tree(true_data, true_labels)
        false_branch = self.build_tree(false_data, false_labels)

        # Возвращаем класс узла со всеми поддеревьями, то есть целого дерева
        return Node(index, t, true_branch, false_branch)

    def classify_object(self, obj, node):

        #  Останавливаем рекурсию, если достигли листа
        if isinstance(node, self._leaf):
            answer = node.prediction
            return answer

        if obj[node.index] <= node.t:
            return self.classify_object(obj, node.true_branch)
        else:
            return self.classify_object(obj, node.false_branch)

    def predict(self, data):

        classes = []
        for obj in data:
            prediction = self.classify_object(obj, self._tree)
            classes.append(prediction)
        return classes

    @staticmethod
    def accuracy_metric(actual, predicted):
        return np.mean(np.where(actual == predicted, 1, False)) * 100.0

        # Напечатаем ход нашего дерева

    def _print_tree(self, node, spacing=""):

        # Если лист, то выводим его прогноз
        if isinstance(node, self._leaf):
            print(spacing + "Прогноз:", node.prediction)
            return

        # Выведем значение индекса и порога на этом узле
        print(spacing + 'Индекс', str(node.index))
        print(spacing + 'Порог', str(node.t))

        # Рекурсионный вызов функции на положительном поддереве
        print(spacing + '--> True:')
        self._print_tree(node.true_branch, spacing + "  ")

        # Рекурсионный вызов функции на положительном поддереве
        print(spacing + '--> False:')
        self._print_tree(node.false_branch, spacing + "  ")

    def print_tree(self):
        return self._print_tree(self._tree)

    def fit(self, data, labels):
        self._tree = self.build_tree(data, labels)
        return self


class DecisionTreeRegressor(TreeBuilder):
    _leaf = RegressorLeaf

    def __init__(self,
                 max_depth: int = None,
                 min_samples_leaf: int = 1,
                 min_samples_split: int = 2,
                 ) -> None:
        super().__init__(max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
        self.criterion = self.variance

    @staticmethod
    def variance(y: np.array) -> float:
        return float(np.std(y))


# In[23]:


class GradientBoostingRegressor:

    def __init__(self,
                 learning_rate: float = 1,
                 n_estimators: int = 10,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 2,
                 max_depth: int = 3) -> None:
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self._trees = None
        self._coefs = [1] * n_estimators
        self.train_errors = None
        self.test_errors = None

    def predict(self,
                X: np.array) -> np.array:
        # Реализуемый алгоритм градиентного бустинга будет инициализироваться нулевыми значениями,
        # поэтому все деревья из списка trees_list уже являются дополнительными и при предсказании прибавляются с шагом eta
        return np.array(
            [sum([self.learning_rate * coef * alg.predict([x])[0] for alg, coef
                  in zip(self._trees, self._coefs)]) for x in X])

    @staticmethod
    def mean_squared_error(y_train, y_pred):
        return np.mean(np.square(y_train - y_pred))

    @staticmethod
    def bias(y, z):
        return y - z

    def fit(self, X_train, X_test, y_train, y_test):
        # Деревья будем записывать в список
        self._trees = []

        # Будем записывать ошибки на обучающей и тестовой выборке на каждой итерации в список
        self.train_errors = []
        self.test_errors = []

        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         min_samples_leaf=self.min_samples_leaf,
                                         min_samples_split=self.min_samples_split)

            # инициализируем бустинг начальным алгоритмом, возвращающим ноль,
            # поэтому первый алгоритм просто обучаем на выборке и добавляем в список
            if not self._trees:
                # обучаем первое дерево на обучающей выборке
                tree.fit(X_train, y_train)

                self.train_errors.append(self.mean_squared_error(y_train, self.predict(X_train)))
                self.test_errors.append(self.mean_squared_error(y_test, self.predict(X_test)))
            else:
                # Получим ответы на текущей композиции
                target = self.predict(X_train)

                # алгоритмы начиная со второго обучаем на сдвиг
                tree.fit(X_train, self.bias(y_train, target))

                self.train_errors.append(self.mean_squared_error(y_train, self.predict(X_train)))
                self.test_errors.append(self.mean_squared_error(y_test, self.predict(X_test)))

            self._trees.append(tree)

        return self

    def evaluate(self, X_train, X_test, y_train, y_test):
        train_prediction = self.predict(X_train)

        print(f'Ошибка алгоритма из {self.n_estimators} деревьев глубиной {self.max_depth}         с шагом {self.learning_rate} на тренировочной выборке: {self.mean_squared_error(y_train, train_prediction)}')

        test_prediction = self.predict(X_test)

        print(f'Ошибка алгоритма из {self.n_estimators} деревьев глубиной {self.max_depth}         с шагом {self.learning_rate} на тестовой выборке: {self.mean_squared_error(y_test, test_prediction)}')

    def error_plot(self):
        plt.xlabel('Iteration number')
        plt.ylabel('MSE')
        plt.xlim(0, self.n_estimators)
        plt.plot(list(range(self.n_estimators)), self.train_errors, label='train error')
        plt.plot(list(range(self.n_estimators)), self.test_errors, label='test error')
        plt.legend(loc='upper right')
        plt.show()


# In[ ]:


tree = GradientBoostingRegressor(learning_rate=0.1,
                                 n_estimators=20,
                                 max_depth=7,
                                 min_samples_leaf=2,
                                 min_samples_split=2)
tree.fit(X_train, X_test, y_train, y_test)
print(tree.evaluate(X_train, X_test, y_train, y_test))
tree.error_plot()


# In[46]:


def r2(y, y_pred):
    return np.abs(1-(mserror(y, y_pred)/np.var(y)))


# In[47]:


r2(labels, test_answers)


# In[48]:


tree_regression.predict(x_final.values)


# In[50]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score

from lightgbm import LGBMRegressor

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[52]:


model = LGBMRegressor(max_depth=5,
                             min_samples_leaf=5,
                             n_estimators=100,
                             random_state=42)

cv_score = cross_val_score(model, x, y, 
                           scoring='r2', 
                           cv=KFold(n_splits=5, shuffle=True, random_state=42))
# cv_score
mean = cv_score.mean()
std = cv_score.std()

print('R2: {:.3f} +- {:.3f}'.format(mean, std))


# In[53]:


# Обучаю модель на всем трейне
model.fit(x, y)


# In[60]:


model.predict(x_final.values)


# In[58]:


preds_final['mean_exam_points'] = tree_regression.predict(x_final.values)
preds_final.to_csv('mark_tree.csv', index=False)


# In[ ]:


preds_final


# In[56]:


tree_regression.predict(x_final.values)


# In[ ]:




