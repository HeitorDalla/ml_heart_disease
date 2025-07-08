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