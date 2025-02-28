import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Definir a semente aleatória para reprodutibilidade
np.random.seed(42)

# =============================================================================
# Parâmetros
# =============================================================================
noiseFactor = 0.0025     # ruído
networkFactor = 100      # para alterar as características da rede (Y)
PtestFactor = 3          # para obter perdas semelhantes aos dados de treino

# =============================================================================
# Leitura dos dados
# =============================================================================
Info = np.array(pd.read_excel(r'DASG_Prob2_new.xlsx', sheet_name='Info', header=None))
SlackBus = Info[0, 1]  # Informação acerca do bus slack

Net_Info = np.array(pd.read_excel(r'DASG_Prob2_new.xlsx', sheet_name='Y_Data'))
Power_Info = np.array(pd.read_excel(r'DASG_Prob2_new.xlsx', sheet_name='Load(t,Bus)'))
Power_Info = np.delete(Power_Info, [0], 1)
Power_Test = np.array(pd.read_excel(r'DASG_Prob2_new.xlsx', sheet_name='Test_Load(t,Bus)'))
Power_Test = np.delete(Power_Test, [0], 1)

time = Power_Info.shape[0]
P = Power_Info
Ptest = Power_Test * PtestFactor

# =============================================================================
# Construção da matriz de admitância Y
# =============================================================================
nBus = int(max(np.max(Net_Info[:, 0]), np.max(Net_Info[:, 1])))
nLines = Net_Info.shape[0]
Y = np.zeros((nBus, nBus), dtype=complex)

for i in range(Net_Info.shape[0]):
    y_aux = Net_Info[i, 2].replace(",", ".").replace("i", "j")
    Y[Net_Info[i, 0]-1, Net_Info[i, 0]-1] += complex(y_aux) * networkFactor
    Y[Net_Info[i, 1]-1, Net_Info[i, 1]-1] += complex(y_aux) * networkFactor
    Y[Net_Info[i, 0]-1, Net_Info[i, 1]-1] -= complex(y_aux) * networkFactor
    Y[Net_Info[i, 1]-1, Net_Info[i, 0]-1] -= complex(y_aux) * networkFactor

# Remover o bus slack da matriz Y
Yl = np.delete(Y, SlackBus-1, axis=0)
Yl = np.delete(Yl, SlackBus-1, axis=1)

G = Yl.real  # Matriz de condutância
B = Yl.imag  # Matriz de susceptância

# =============================================================================
# Construção da matriz de incidência C
# =============================================================================
C = np.zeros((nBus, nLines))
nLine_Aux = 0
for i in range(nBus):
    for j in range(i+1, nBus):
        if abs(Y[i, j]) != 0:
            C[i, nLine_Aux] = 1
            C[j, nLine_Aux] = -1
            nLine_Aux += 1

# Remover o bus slack da matriz C para obter Cl
Cl = np.delete(C, SlackBus-1, axis=0)

# =============================================================================
# Construção dos vetores de condutância das linhas
# =============================================================================
Gv = np.zeros((1, nLines))
Gd = np.zeros((nLines, nLines))
nLine_Aux = 0
for i in range(nBus):
    for j in range(i+1, nBus):
        if abs(Y[i, j]) != 0:
            Gv[0, nLine_Aux] = -np.real(Y[i, j])
            Gd[nLine_Aux, nLine_Aux] = -np.real(Y[i, j])
            nLine_Aux += 1

# =============================================================================
# Função para calcular as perdas físicas (PL2) utilizando a Eq (13)
# =============================================================================
def compute_PL2(P_mat, B, Cl, Gv, noiseFactor):
    """
    Calcula as perdas físicas (PL2) para cada instante de tempo utilizando a Eq (13).

    Parâmetros:
      P_mat: Matriz de potência, com dimensão (n_instantes, nBus)  (ou nBus-1 se o slack for removido)
      B: Matriz de susceptância, dimensão (nBus-1, nBus-1)
      Cl: Matriz de incidência (com slack removido), dimensão (nBus-1, nLines)
      Gv: Vetor de condutância das linhas, dimensão (1, nLines)
      noiseFactor: Fator de ruído multiplicativo

    Retorna:
      PL2_vec: Vetor das perdas para cada instante (valores escalares)
    """
    n_instantes = P_mat.shape[0]
    PL2_vec = np.zeros(n_instantes)
    invB = inv(B)  # Pré-computa a inversa de B
    for m in range(n_instantes):
        # Calcular os ângulos de tensão: teta = inv(B) @ P_mat[m, :].T
        teta = invB @ P_mat[m, :].T
        # Calcular as diferenças dos ângulos: grau = Cl.T @ teta
        grau = Cl.T @ teta
        # Calcular as perdas físicas: PL2 = 2 * Gv @ (1 - cos(grau))
        PL2_m = (2 * Gv) @ (1 - np.cos(grau))
        # Extrair o valor escalar a partir do array 1x1
        PL2_scalar = PL2_m.item()
        # Adicionar ruído multiplicativo
        PL2_vec[m] = PL2_scalar * (1 + np.random.normal(0, noiseFactor))
    return PL2_vec

# =============================================================================
# Cálculo das perdas físicas para os dados de treino (PL2)
# =============================================================================
PL2_train_true = compute_PL2(P, B, Cl, Gv, noiseFactor)

# =============================================================================
# Cálculo da matriz X para os dados de treino (termos de segundo grau)
# =============================================================================
X = np.column_stack((
    P[:, 0]**2,              # P1^2
    2 * P[:, 0] * P[:, 1],   # 2*P1*P2
    2 * P[:, 0] * P[:, 2],   # 2*P1*P3
    2 * P[:, 0] * P[:, 3],   # 2*P1*P4
    P[:, 1]**2,              # P2^2
    2 * P[:, 1] * P[:, 2],   # 2*P2*P3
    2 * P[:, 1] * P[:, 3],   # 2*P2*P4
    P[:, 2]**2,              # P3^2
    2 * P[:, 2] * P[:, 3],   # 2*P3*P4
    P[:, 3]**2               # P4^2
))

# =============================================================================
# Cálculo do novo beta utilizando OLS
# beta_novo = (X^T X)^{-1} X^T * y, onde y é PL2_train_true
# =============================================================================
beta_novo = inv(X.T @ X) @ (X.T @ PL2_train_true)

# Predição para os dados de treino
PL2_train_pred = X @ beta_novo

# Cálculo das métricas de erro para o treino
train_rmse = np.sqrt(mean_squared_error(PL2_train_true, PL2_train_pred))
train_mae = mean_absolute_error(PL2_train_true, PL2_train_pred)

print("Erro de treino - RMSE:", train_rmse)
print("Erro de treino - MAE: ", train_mae)

# =============================================================================
# Construção da matriz X_test para os dados de teste (mesmos termos de segundo grau)
# =============================================================================
X_test = np.column_stack((
    Ptest[:, 0]**2,              # P1^2
    2 * Ptest[:, 0] * Ptest[:, 1],   # 2*P1*P2
    2 * Ptest[:, 0] * Ptest[:, 2],   # 2*P1*P3
    2 * Ptest[:, 0] * Ptest[:, 3],   # 2*P1*P4
    Ptest[:, 1]**2,              # P2^2
    2 * Ptest[:, 1] * Ptest[:, 2],   # 2*P2*P3
    2 * Ptest[:, 1] * Ptest[:, 3],   # 2*P2*P4
    Ptest[:, 2]**2,              # P3^2
    2 * Ptest[:, 2] * Ptest[:, 3],   # 2*P3*P4
    Ptest[:, 3]**2               # P4^2
))

# Predição para os dados de teste utilizando o modelo de regressão
PL2_test_pred = X_test @ beta_novo

# =============================================================================
# Cálculo das perdas físicas para os dados de teste (utilizando a mesma função)
# =============================================================================
PL2_test_true = compute_PL2(Ptest, B, Cl, Gv, noiseFactor)

# =============================================================================
# Gráfico das comparações (Treino e Teste) utilizando gráficos em escadas (step plots) em subplots
# =============================================================================
time_intervals_train = np.arange(len(PL2_train_true))
time_intervals_test = np.arange(len(PL2_test_true))

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

# Gráfico para o conjunto de treino
ax[0].step(time_intervals_train, PL2_train_true, where='post', label='Treino: Perdas Reais (PL2 com ruído)')
ax[0].step(time_intervals_train, PL2_train_pred, where='post', label='Treino: Perdas Preditas', linestyle='--')
ax[0].set_title('Comparação de Perdas - Treino')
ax[0].set_xlabel('Carimbo Temporal [passo temporal]')
ax[0].set_ylabel('Perdas de Potência')
ax[0].legend(loc='upper right')
ax[0].grid(True)

# Gráfico para o conjunto de teste
ax[1].step(time_intervals_test, PL2_test_true, where='post', label='Teste: Perdas Reais (Física)', marker='o')
ax[1].step(time_intervals_test, PL2_test_pred, where='post', label='Teste: Perdas Preditas (Regressão)', marker='s', linestyle='--', color='orange')
ax[1].set_title('Comparação de Perdas para Dados de Teste')
ax[1].set_xlabel('Carimbo Temporal [passo temporal]')
ax[1].set_ylabel('Perdas de Potência')
ax[1].legend(loc='upper right')
ax[1].grid(True)

plt.tight_layout()
plt.show()

# =============================================================================
# Cálculo das métricas de erro para o conjunto de teste
# =============================================================================
test_rmse = np.sqrt(mean_squared_error(PL2_test_true, PL2_test_pred))
test_mae = mean_absolute_error(PL2_test_true, PL2_test_pred)

print("Erro de teste - RMSE:", test_rmse)
print("Erro de teste - MAE: ", test_mae)
