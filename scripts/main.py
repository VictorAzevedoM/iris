import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Carregar o conjunto de dados Iris
iris = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')

# Visualizar as primeiras linhas do conjunto de dados
print(iris.head())

# Resumo estatístico das características
print(iris.describe())

# Contagem de amostras por espécie
print(iris['species'].value_counts())


# Gráfico de dispersão das características por espécie
sns.pairplot(iris, hue='species')
plt.show()

# Dividir os dados em conjunto de recursos (X) e rótulos (y)
X = iris.drop('species', axis=1)
y = iris['species']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar o classificador KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Treinar o modelo com os dados de treinamento
knn.fit(X_train, y_train)

# Prever os rótulos para os dados de teste
predictions = knn.predict(X_test)

# Avaliar a precisão do modelo
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

