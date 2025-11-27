import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gc

df = pd.read_csv('result.csv', sep=';', usecols=[
    'gaming_interest_score',
    'insta_design_interest_score',
    'football_interest_score',
    'recommended_product',
    'campaign_success',
    'age',
    'canal_recommande'
])

# nombre de valeurs nulles 
print(df.isnull().sum())
df = df.dropna()

# nombre des valeurs doublons
# print(df.duplicated().sum())


# print(df.dtypes)

# df['Id'] = df['Id'].astype('int16') # passage de 1O.5kb à 9.5kb
df['age'] = df['age'].astype('int8')
df['gaming_interest_score'] = df['gaming_interest_score'].astype('int16')
df['insta_design_interest_score'] = df['insta_design_interest_score'].astype('int16')
df['football_interest_score'] = df['football_interest_score'].astype('int16')
df['football_interest_score'] = df['football_interest_score'].astype('int16')

df['recommended_product'] = df['recommended_product'].str.strip().str.lower().astype('category')
df['campaign_success'] = df['campaign_success'].astype('bool')
df['canal_recommande'] = df['canal_recommande'].str.strip().str.lower().astype('category')




# get types of columns
print(df.info())
print(df.head())
del df
gc.collect()