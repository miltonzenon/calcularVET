import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 1. Gerando Dataset com as variáveis do artigo de Ponce et al.
np.random.seed(42)
n_pacientes = 500

data = {
    'idade': np.random.randint(18, 85, n_pacientes),
    'peso': np.random.uniform(50, 120, n_pacientes),
    'altura': np.random.uniform(150, 195, n_pacientes),
    'sexo': np.random.choice([0, 1], n_pacientes),
    'fc': np.random.uniform(60, 120, n_pacientes),
    'temp': np.random.uniform(36, 39.5, n_pacientes),
    'pcr': np.random.uniform(0.5, 150, n_pacientes),
    # Novas Variáveis
    'ureia': np.random.uniform(20, 200, n_pacientes),
    'creatinina': np.random.uniform(0.6, 5.0, n_pacientes),
    'ph_arterial': np.random.uniform(7.20, 7.45, n_pacientes),
    'volume_minuto': np.random.uniform(3, 15, n_pacientes), # Opcional (0 se não ventilado)
    'peep': np.random.uniform(5, 15, n_pacientes)           # Opcional (0 se não ventilado)
}

df = pd.DataFrame(data)

# Lógica de cálculo do Target (inspirada no r=0.69 do artigo)
# Note que MV e PEEP agora têm um peso grande no cálculo
df['geb_real'] = (10 * df['peso']) + (6.25 * df['altura']) - (5 * df['idade'])
df['geb_real'] += (df['volume_minuto'] * 45) + (df['peep'] * 20) # Impacto ventilatório
df['geb_real'] += (df['ureia'] * 0.5) + (df['temp'] - 37).clip(lower=0) * 150
df['geb_real'] += np.random.normal(0, 30, n_pacientes)

# 2. Treinando o Modelo
X = df.drop('geb_real', axis=1)
y = df['geb_real']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

modelo_final = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6)
modelo_final.fit(X_train, y_train)

print("Cérebro treinado com sucesso com variáveis de alta complexidade!")
