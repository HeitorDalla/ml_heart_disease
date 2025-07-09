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

df['target'].value_counts().plot(kind="bar",
                                 color=["salmon", "lightblue"])
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

comparation_sex_target.plot(kind='bar',
                            figsize=(10,6),
                            color=['Salmon', 'lightblue'])
plt.title('Frequência entre Masculino e Feminino')
plt.xlabel("0 - Sem doença | 1 - Com doença")
plt.ylabel("Quantidade")
plt.legend(['Feminino', 'Masculino'])
plt.show()

# Comparação entre as variáveis independentes thalach (frequência máxima cardiaca) e age (idade), 
# e a variável dependente target (tem ou não tem doença)

# Criar outra figura
plt.figure(figsize=(10, 6))

# Fazer um gráfico de disperção onde o target = 1, ou seja, quando são positivos para a doença
plt.scatter(df.loc[df['target'] == 1, 'age'],
            df.loc[df['target'] == 1, 'thalach'],
            c='red')

# Fazer um gráfico de disperção onde o target = 0, ou seja, quando são falsos para a doença
plt.scatter(df.loc[df['target'] == 0, 'age'],
            df.loc[df['target'] == 0, 'thalach'],
            c='blue')

# Análise dos gráficos - quanto mais jovens, maior a frequência cardiaca

# Configurações de personalizações
plt.title("Disperção da doença em relação à Idade e Frequência Cardíaca Máxima")
plt.xlabel("Idade")
plt.ylabel("Frequência Cardíaca Máxima")
plt.legend(['1 - Doença', '0 - Não Doença'])
plt.show()

# Pegar os outliers
min_thalach = df['thalach'].min()
outlier = df[df['thalach'] == min_thalach] # pessoa que possui menor frequência cardíaca máxima
indice_outliers = outlier.index # pegar os indices
df_sem_outliers = df.drop(indice_outliers, axis=0)

# Vendo a distribuição de Idades para verificar outliers
df['age'].plot(kind='hist')
plt.show()

# Relações entre cp (dor no peito) e ter doença ou não
comparation_cp_target = pd.crosstab(df['cp'], df['target'])
print(comparation_cp_target)

comparation_cp_target.plot(kind='bar',
                           figsize=(10, 6),
                           color=['Red', 'Blue'])

# Configurações de Personalização
plt.title("Frequência dos tipos de dores no peito com a doença")
plt.xlabel("Tipos de dores no peito")
plt.ylabel("Quantidade")
plt.legend(['Sem doença', 'Com doença'])
plt.show()

# Construindo uma Matriz de Correlação entre as variáveis
print("Matriz de Correlação")
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(df.corr(),
                 annot=True, # indica que vai mostras o valor numérico dentro do quadrados (não somente as cores)
                 linewidths=0.5, # controla a largura entre as linhas que separam as células do mapa de calor (para remover, basta colocar)
                 fmt=".2f", # defini o formato que os números vão aparecer, no caso, com 2 casas decimais
                 cmap="YlGnBu") # defini a paleta de cores, no caso, yellow, green e blue
plt.yticks(rotation=45)
plt.show()