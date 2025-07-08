# Estruturação dos dados
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Modelo de algorítmos
from sklearn.ensemble import RandomForestClassifier

# Avaliação do modelo
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# Melhorar os hiperparâmetros
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Salvar um modelo
import pickle


# Tratamento dos Dados

# Dicionário de Dados


# Que pergunta(s) você está tentando resolver?
# Que tipo de dados temos e como tratamos os diferentes tipos?
# O que está faltando nos dados e como você lida com isso?
# Onde estão os valores outliers e por que você deveria se importar com eles?
# Como você pode adicionar, alterar ou remover recursos para aproveitar melhor seus dados?

df = pd.read_csv('datA/heart-disease.csv', sep=',', encoding='utf-8')
print(df)

# Linhas x colunas
print(df.shape) 

# Primeiras linhas
print(df.head())

# Ver a quantidade de cada valor na coluna alvo
print(df['target'].value_counts())

df['target'].value_counts().plot(kind="bar", color=["salmon", "lightblue"])
plt.show()

# Vendo demais informações sobre o dataset
print(df.describe())

print(df.info())

print(df.isna().sum())

# Comparação de colunas entre si para poder achar padrões

# Sex and target
print(df['sex'].value_counts()) # muito mais masculino do que feminino

comparation_sex_target = pd.crosstab(df['target'], df['sex'])
print(comparation_sex_target)