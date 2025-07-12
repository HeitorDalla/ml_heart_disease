# Estruturação dos dados
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Modelo de algorítmos
from sklearn.linear_model import LogisticRegression # muito utilizado quando tem fortes relações lineares entre as variáveis
from sklearn.ensemble import RandomForestClassifier # muito utilizado quando tem relações lineares e não lineares
from sklearn.ensemble import GradientBoostingClassifier # muito utilizado quando tem fortes relações não lineares entre as variáveis

# Avaliação do modelo
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report, RocCurveDisplay

# Melhorar os hiperparâmetros
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Salvar um modelo
import pickle

# Pipeline para o LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


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

# Dividindo os dados em variáveis dependentes e independentes
X = df_sem_outliers.drop(columns='target', axis=1)
y = df_sem_outliers['target']

# Separação dos dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Escolhendo o modelo para utilizar
models = {
    'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting Classifier': GradientBoostingClassifier()
}

# Função que vai retornar um dicionário que vai conter o nome do modelo e seu score
def fit_and_score(models, X_train, X_test, y_train, y_test):
    np.random.seed(42)

    # Dicionário para manter os scores dos modelos
    models_scores = {}

    for name_module, module in models.items():
        # Treinar o modelo
        training = module.fit(X_train, y_train)

        # Avaliar e salvar o modelo no dicionário
        models_scores[name_module] = module.score(X_test, y_test)

    # Retornar o dicionário dos scores dos modelos
    return models_scores

# Dicionário que vai armazenar o modelo e seu score
models_score = fit_and_score(models=models,
                             X_train=X_train,
                             X_test=X_test,
                             y_train=y_train,
                             y_test=y_test)

# Modelos e scores
print("Modelos e seus scores: ")
for name, score in models_score.items():
    print("{} - {}" .format(name, score))

# Logistic Regression - 0.8852459016393442
# Random Forest - 0.8688524590163934
# Gradient Boosting Classifier - 0.7540983606557377

# Comparação dos Modelos
pd.DataFrame(models_score, index=['accuracy']).T.plot(kind='bar')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()


# Ajustes de Hiperparâmetros

# RandomizedSearchCV - Pela quantidade de vezes que eu definir, ele vai fazer validação cruzada com os dados


# LogisticRegression

# Definindo os Hiperparâmetros modificados
rs_lr_grid = {"C": np.logspace(-4, 4, 20),
              "solver": ["liblinear"]}

# Garantia de Reprodutibilidade
np.random.seed(42)

# Instanciando o modelo (objeto de busca por hiperparâmetros)
rs_lr_model = RandomizedSearchCV(LogisticRegression(max_iter=10000),
                                 param_distributions=rs_lr_grid,
                                 n_iter=20,
                                 cv=5,
                                 verbose=2)

# Treinamento do Modelo
rs_lr_training = rs_lr_model.fit(X_train, y_train)

# Calcula as 20 combinações aleatórias dos hiperparâmetros, 
# para cada combinação, realiza a validação cruzada com 5 fols, 
# onde cada fold usa 80% para treino e 20% para testes (diferente para cada rodada), e para finalizar,
# calcula a acurácia média de cada combinação dos 5 folds
# Escolhe o melhor modelo (com maior acurácia média)
# Conteste o modelo final com os melhores hiperparâmetros usando todo o X_train
# O modelo final esta pronto.

# Ja treinado, agora o 'rs_lr_model' (modelo) possui o melhor resultado
rs_lr_params = rs_lr_model.best_params_
print("Os melhores hiperparâmetros do modelo Logistic Regression potencializado pelo Randomized Search são: ")
print(rs_lr_params)

# Pega o melhor modelo final treinado com os melhor hiperparâmetros usando todo o X_train e y_train
rs_lr_estimator = rs_lr_model.best_estimator_
print("O melhor modelo final do Logistic Regression potencializado pelo Randomized Search são: ")
print(rs_lr_estimator)

# É a acurácia média dos 5 volds (da melhor combinação) do best_estimator (melhor modelo, treinado com os melhores parâmetros (best_params_))
rs_lr_score = rs_lr_model.best_score_
print("A acurácia média do melhor modelo Logistic Regression potencializado pelo Randomized Search são: ")
print(rs_lr_score)
# 0.8176020408163265

# Avaliação do Modelo - Avalia o best_estimator em dados que ele nunca viu (X_test, y_test)
rs_lr_accuracy = rs_lr_model.score(X_test, y_test)
print("A avaliação do modelo Logistic Regression treinado pelo Randomized Search é: {}" .format(rs_lr_accuracy))
# 0.868852459016393


# RandomForestClassifier
rs_rf_grid = {
    "max_depth": [None, 5, 10, 20, 30],
    "max_features": ['sqrt', 'log2'],
    "min_samples_leaf": [1, 2, 4],
    "min_samples_split": [2, 4, 6],
    "n_estimators": [10, 100, 200, 500]
}

# Garantia de Reprodutibilidade
np.random.seed(42)

# Instancia do Modelo
rs_rf_model = RandomizedSearchCV(RandomForestClassifier(),
                                 param_distributions=rs_rf_grid,
                                 n_iter=20,
                                 cv=5,
                                 verbose=2)

# Treinar o modelo
rs_rf_training = rs_rf_model.fit(X_train, y_train)

# Pegar os melhores Hiperparâmetros
rs_rf_params = rs_rf_model.best_params_
print("Os melhores hiperparâmetros do modelo Random Forest Classifier potencializado pelo Randomized Search são: ")
print(rs_rf_params) # {'n_estimators': 10, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 5}

# Avaliação do Modelo
rs_rf_score = rs_rf_model.score(X_test, y_test)
print("A avaliação do modelo Random Forest Classifier treinado pelo Randomized Search é: {}" .format(rs_rf_score))
# 0.8688524590163934


# GridSearchCV - Com os Hiperparâmetros que eu passar, ele vai fazer todas as opções que tem com validação cruzada


# LogisticRegression

# Inicializar os Hiperparâmetros a mudar
gs_lr_grid = {"C": np.logspace(-4, 4, 30),
              "solver": ["liblinear"]}

# Garantia de Reprodutibilidade
np.random.seed(42)

# Inicializar o modelo
gs_lr_model = GridSearchCV(LogisticRegression(max_iter=10000),
                           param_grid=gs_lr_grid,
                           cv=5,
                           verbose=2)

# Treinamento do Modelo
gs_lr_training = gs_lr_model.fit(X_train, y_train)

# Pegar os melhores Hiperparâmetros do treinamento
gs_lr_params = gs_lr_model.best_params_
print("Os melhores hiperparâmetros do modelo Logistic Regression potencializado pelo Grid Search são: ")
print(gs_lr_params)

# Avaliação do Modelo
gs_lr_score = gs_lr_model.score(X_test, y_test)
print("A avaliação do modelo Logistic Regression treinado pelo Grid Search é: {}" .format(gs_lr_score))
# 0.8524590163934426


# RandomForestClassifier
gs_rf_grid = {'n_estimators': [100, 200, 500],
              'min_samples_split': [4],
              'min_samples_leaf': [4],
              'max_features': ['sqrt'],
              'max_depth': [None]}

# Garantia de Reprodutibilidade
np.random.seed(42)

# Instanciando o Modelo
gs_rf_model = GridSearchCV(RandomForestClassifier(),
                           param_grid=gs_rf_grid,
                           cv=5,
                           verbose=2)

# Treinamento do Modeo
gs_rf_training = gs_rf_model.fit(X_train, y_train)

# Melhores Hiperparâmetros
gs_rf_params = gs_rf_model.best_params_
print("Os melhores hiperparâmetros do modelo Random Forest Classifier potencializado pelo Grid Search são: ")
print(gs_rf_params)

# Avaliando o Modelo treinado
gs_rf_score = gs_rf_model.score(X_test, y_test)
print("A avaliação do modelo Random Forest Classifier treinado pelo Grid Search é: {}" .format(gs_rf_score))
# 0.8524590163934426


# Escolhendo o melhor modelo até agora e fazendo previsões
best_params_model = rs_rf_model.best_params_ # {'n_estimators': 10, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 5}

best_estimator_model = rs_rf_model.best_estimator_ # melhor modelo já treinado

# Originando um novo modelo com os melhor parâmetros
clf = RandomForestClassifier(n_estimators = 10, 
                             min_samples_split = 4, 
                             min_samples_leaf = 2, 
                             max_features = 'log2', 
                             max_depth = 5)

y_preds = rs_rf_model.predict(X_test)


# Confusion Matrix - 
print(confusion_matrix(y_test, y_preds))

# Gráfico
sns.set_theme(font_scale=1.5)

fig, ax = plt.subplots(figsize=(3, 3))
ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                 annot=True,
                 cbar=False)
plt.xlabel("Verdadeiros rótulos")
plt.ylabel("Previsão dos rótulos")
plt.show()


# Roc_curve and Roc_auc_score
RocCurveDisplay.from_estimator(best_estimator_model, X_test, y_test)
plt.title("Curva ROC - Melhor Modelo Random Forest")
plt.show()


# Classification report - 
print(classification_report(y_test, y_preds))


# Cross-val-score

# Accuracy - Tem comm função ver a acurácia do modelo
cv_accuracy = cross_val_score(clf,
                              X,
                              y,
                              cv=5,
                              scoring='accuracy')
cv_accuracy_mean = np.mean(cv_accuracy)

print("A acurácia do melhor modelo é: {}" .format(cv_accuracy_mean))

# Precision - Tem como função ver a precisão do modelo
cv_precision = cross_val_score(clf,
                              X,
                              y,
                              cv=5,
                              scoring='precision')
cv_precision_mean = np.mean(cv_precision)

print("A precisão do melhor modelo é: {}" .format(cv_precision_mean))

# Recall - 
cv_recall = cross_val_score(clf,
                            X,
                            y,
                            cv=5,
                            scoring='recall')
cv_recall_mean = np.mean(cv_recall)

print("O recall do melhor modelo é: {}" .format(cv_recall_mean))

# F1 - 
cv_f1 = cross_val_score(clf,
                        X,
                        y,
                        cv=5,
                        scoring='f1')
cv_f1_mean = np.mean(cv_f1)

print("O f1-score do melhor modelo é: {}" .format(cv_f1_mean))

# Comparação das avaliações do Cross-val-score
cv_metrics = pd.DataFrame({
    'Accuracy': cv_accuracy_mean,
    'Precision': cv_precision_mean,
    'Recall': cv_recall_mean,
    'F1': cv_f1_mean
}, index=[0])

cv_metrics.T.plot.bar(title="Classificação das métricas da Validação Cruzada",
                      legend=False)
plt.xticks(rotation=50)
plt.show()