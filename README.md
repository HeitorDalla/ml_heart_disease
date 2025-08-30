# 🫀 Predição de Doença Cardíaca

Este projeto tem como objetivo principal **construir e avaliar um modelo de Machine Learning** capaz de prever a presença de doenças cardíacas em pacientes, com base em diversos atributos clínicos.  
O projeto segue um **pipeline completo de Machine Learning**, desde a **análise exploratória dos dados (EDA)** e **pré-processamento**, até a **seleção do melhor modelo**, **otimização de hiperparâmetros** e **avaliação final**.

---

## 🎯 Objetivo
O principal objetivo deste projeto é desenvolver um modelo preditivo robusto para classificar pacientes em duas categorias:

- **1** → Paciente com doença cardíaca
- **0** → Paciente sem doença cardíaca

---

## 🚀 Tecnologias Utilizadas

- **Linguagem:** Python
- **Bibliotecas:**  
  - `pandas` e `numpy` → Manipulação e análise de dados  
  - `matplotlib` e `seaborn` → Visualização de dados e gráficos  
  - `scikit-learn` → Construção, avaliação e otimização dos modelos  
  - `pickle` → Serialização e salvamento do modelo treinado

---

## 📊 Estrutura do Projeto

```
📂 ml_heart_disease
├── 📁 data                # Contém o dataset
│    └── heart-disease.csv
├── 📁 models              # Pasta para salvar o modelo treinado
│    └── best_model_rf.pkl
├── main.py                # Script principal com todo o pipeline de ML
└── README.md              # Documentação do projeto
```

---

## 📁 Dataset

O dataset **heart-disease.csv** contém **14 atributos clínicos** dos pacientes. Alguns deles:

- **age** → Idade  
- **sex** → Gênero (1 = masculino, 0 = feminino)  
- **cp** → Tipo de dor no peito  
- **thalach** → Frequência cardíaca máxima alcançada  
- **target** → Indica presença (1) ou ausência (0) de doença cardíaca

---

## ⚙️ Pipeline de Machine Learning

O script **main.py** executa as seguintes etapas:

### **1. Análise Exploratória e Pré-processamento**
- Carregamento e limpeza dos dados
- Análise visual com gráficos de barra, dispersão e distribuição
- Remoção de **outliers** (por exemplo, valores anômalos em `thalach`)
- Geração da **matriz de correlação** para entender relações entre atributos

### **2. Modelagem e Treinamento**
- Divisão dos dados → **80% treino / 20% teste**
- Avaliação de 3 modelos:
  - **Logistic Regression**
  - **Random Forest Classifier**
  - **Gradient Boosting Classifier**
- Comparação inicial de acurácia

### **3. Otimização de Hiperparâmetros**
- **RandomizedSearchCV** → Busca aleatória por melhores hiperparâmetros  
- **GridSearchCV** → Busca exaustiva para refinar os parâmetros

### **4. Avaliação Final**
- Seleção do **melhor modelo** (Random Forest otimizado)
- **Matriz de confusão** + **Heatmap**
- **Curva ROC** + **AUC**
- **Classification Report** → Precision, Recall e F1-score
- **Cross-validation** para validação robusta

### **5. Análise de Importância das Features**
- Uso de **Permutation Importance** para verificar quais atributos são mais relevantes

### **6. Salvamento do Modelo**
- O modelo final é salvo como **best_model_rf.pkl** usando `pickle`

---

## 🛠️ Como Executar o Projeto

### **1. Clone o repositório**
```bash
git clone https://github.com/HeitorDalla/ml_heart_disease
cd ml_heart_disease
```

### **2. Crie e ative o ambiente virtual**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### **3. Instale as dependências**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### **4. Certifique-se de que o dataset está na pasta `/data`**

### **5. Execute o script principal**
```bash
python main.py
```

O modelo treinado será salvo automaticamente na pasta `/models`.

---

## 🤝 Contribuições

Contribuições são bem-vindas!  
Caso encontre bugs ou tenha sugestões de melhorias, **abra uma issue** ou envie um **pull request**.

---

- 📌 **Autor:** Heitor Giussani Dalla Villa
- 📧 **Contato:** heitorvillavilla@email.com  
