#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[5]:


from sklearn.preprocessing import KBinsDiscretizer


# In[6]:


# Sua análise começa aqui.
countries.describe()


# In[7]:


countries.info()


# In[8]:


columns_to_float = [
    "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service", 'Climate'
]


# In[9]:


countries2 = countries.copy()


# In[10]:


countries2.Region = countries2.Region.apply(lambda x: x.strip())
countries2.Country = countries2.Country.apply(lambda x: x.strip())


# In[11]:


for c in columns_to_float:
    notnas = countries2[c].notna()
    countries2[c].loc[notnas] = countries2[c].loc[notnas].apply(lambda x: np.float(x.replace(',','.')))
#    countries2[c] = countries2[c].fillna(countries2[c].loc[notnas].median())


# In[12]:


#notnas = countries2.GDP.notna()
#countries2.GDP = countries2.GDP.fillna(countries2.GDP.loc[notnas].median())


# In[13]:


#countries2.head()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[14]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return sorted(countries2.Region.unique().tolist())
q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[15]:


def q2():
    # Retorne aqui o resultado da questão 2.
    kbd = sk.preprocessing.KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    kbd.fit(countries2.Pop_density.values.reshape(-1,1))

    x = kbd.transform(countries2.Pop_density.values.reshape(-1,1))
    return int(sum(x > np.quantile(x, 0.9))[0])
q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[16]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return countries2.Climate.unique().shape[0] + countries2.Region.unique().shape[0]
q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[17]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[18]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# In[19]:


def q4():
    # Retorne aqui o resultado da questão 4.
    df_test_country = pd.DataFrame([test_country], columns=countries2.columns)
    pipe_line = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
        ('Scaler', StandardScaler())
    ])
    colmuns_test = df_test_country.drop(columns=['Country', 'Region'], axis=1).columns
    pipe_line.fit(countries2[colmuns_test])
    df_test_country[colmuns_test] = pipe_line.transform(df_test_country[colmuns_test])
    return float(df_test_country.Arable[0].round(3))
q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[20]:


def q5():
    # Retorne aqui o resultado da questão 4.
    net_migration = countries2.Net_migration.dropna()
    q1 = np.quantile(net_migration, .25)
    q3 = np.quantile(net_migration, .75)
    iqr = q3 - q1
    inferior = q1 - 1.5 * iqr
    superior = q3 + 1.5 * iqr
    qtd_i = int((net_migration < inferior).sum())
    qtd_s = int((net_migration > superior).sum())
    print((qtd_i + qtd_s) / net_migration.shape[0] * 100)
    return (qtd_i, qtd_s, False)
q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[21]:


from sklearn.datasets import fetch_20newsgroups


# In[22]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[23]:


from sklearn.feature_extraction.text import CountVectorizer


# In[24]:


def q6():
    # Retorne aqui o resultado da questão 4.
    cv = CountVectorizer()
    vector_count = cv.fit_transform(newsgroup.data)
    index_phone = cv.vocabulary_.get('phone')
    return int(vector_count.toarray()[:,index_phone].sum())
q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[25]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[26]:


def q7():
    # Retorne aqui o resultado da questão 4.
    tv = TfidfVectorizer()
    vector_count_tv = tv.fit_transform(newsgroup.data)
    index_phone = tv.vocabulary_.get('phone')
    return float(vector_count_tv.getcol(index_phone).sum().round(3))
q7()


# In[ ]:




