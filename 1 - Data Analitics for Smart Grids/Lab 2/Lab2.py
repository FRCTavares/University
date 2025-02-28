###############################################################################################
# Laboratorio 2 - Power Losses                                                                #
#                                                                                             #
# Grupo X                                                                                     #
#                                                                                             #
# Membros:                                                                                    #
#   Francisco Tavares (103402)                                                                #
#   Marta Valente (103574)                                                                    #
###############################################################################################

import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

noiseFactor=0.0025     # ruído
networkFactor=100      # para alterar as características da rede (Y)
PtestFactor=3          # para obter perdas semelhantes aos dados de treino

Info = np.array(pd.read_excel (r'DASG_Prob2_new.xlsx', sheet_name='Info', header=None))

# Informação acerca do bus slack
SlackBus=Info[0,1]
# print ("Slack Bus: ", SlackBus,"\n")

# Informação da rede
Net_Info = np.array(pd.read_excel (r'DASG_Prob2_new.xlsx', sheet_name='Y_Data'))
# print ("Informações das linhas (Admitâncias)\n", Net_Info, "\n")

# Informação de Potência (treino)
Power_Info = np.array(pd.read_excel (r'DASG_Prob2_new.xlsx', sheet_name='Load(t,Bus)'))
Power_Info = np.delete(Power_Info,[0],1)
# print ("Informação de consumo de potência (tempo, Bus) - (Treino)\n", Power_Info, "\n")

# Informação de Potência (teste)
Power_Test = np.array(pd.read_excel (r'DASG_Prob2_new.xlsx', sheet_name='Test_Load(t,Bus)'))
Power_Test = np.delete(Power_Test,[0],1)
# print ("Informação de consumo de potência (tempo, Bus) - (Teste)\n", Power_Test)

time=Power_Info.shape[0]
P=Power_Info
Ptest=Power_Test *PtestFactor

# Determinar o número de buses
nBus=max(np.max(Net_Info[:,0]),np.max(Net_Info[:,1]))

# Criar a variável número de linhas e a matriz de admitâncias (Y)
nLines=Net_Info.shape[0]

Y=np.zeros((nBus,nBus), dtype=complex)

# Completar a matriz Y e atualizar o número de linhas
for i in range (Net_Info.shape[0]):
    y_aux=Net_Info[i,2].replace(",",".")
    y_aux=y_aux.replace("i","j")
    Y[Net_Info[i,0]-1,Net_Info[i,0]-1]=Y[Net_Info[i,0]-1,Net_Info[i,0]-1]+complex(y_aux)*networkFactor
    Y[Net_Info[i,1]-1,Net_Info[i,1]-1]=Y[Net_Info[i,1]-1,Net_Info[i,1]-1]+complex(y_aux)*networkFactor
    Y[Net_Info[i,0]-1,Net_Info[i,1]-1]=Y[Net_Info[i,0]-1,Net_Info[i,1]-1]-complex(y_aux)*networkFactor
    Y[Net_Info[i,1]-1,Net_Info[i,0]-1]=Y[Net_Info[i,1]-1,Net_Info[i,0]-1]-complex(y_aux)*networkFactor

            
# Remover o bus slack da matriz de admitâncias
Yl=np.delete(Y, np.s_[SlackBus-1], axis=0)
Yl=np.delete(Yl, np.s_[SlackBus-1], axis=1)


# Matriz de condutância
G=Yl.real

# Matriz de susceptância
B=Yl.imag 
# print("A matriz de admitância Y é:\n", Y, "\n")
# print("A matriz de condutância G é:\n", G, "\n")
# print("A matriz de susceptância B é:\n",B, "\n")

# Criar os vetores
C=np.zeros((nBus,nLines))
nLine_Aux=0

# Determinar a Matriz de Incidência
for i in range (Y.shape[0]):
    for j in range (i+1,Y.shape[1]):
        if np.absolute(Y[i,j])!=0:
            C[i,nLine_Aux]=1
            C[j,nLine_Aux]=-1
            nLine_Aux=nLine_Aux+1           

# Remover o bus slack da matriz
Cl=np.delete(C, np.s_[SlackBus-1], axis=0)

# print ("A matriz de incidência C (nBus,nLines) é:\n",Cl)

# Criar os vetores
Gv=np.zeros((1,nLines))
Gd=np.zeros((nLines,nLines))
nLine_Aux=0

# Determinar a Matriz de Incidência
for i in range (Y.shape[0]):
    for j in range (i+1,Y.shape[1]):
        if np.absolute(Y[i,j])!=0:
            Gv[0,nLine_Aux]=-np.real(Y[i,j])          # Informação acerca das condutâncias das linhas [Vetor]
            Gd[nLine_Aux,nLine_Aux]=-np.real(Y[i,j])  # Informação acerca das condutâncias das linhas [Diagonal em matriz]
            nLine_Aux=nLine_Aux+1           

# print ("Gij_Diag:\n",Gd)

# Criação da matriz
teta=np.zeros((nBus-1,time))
grau=np.zeros((nLines,time))
PL2=np.zeros((time))
PT=np.zeros((time))
rLoss=np.zeros((time))

# Perdas
alfa=np.dot(np.dot(np.dot(np.dot(np.linalg.inv(B),Cl),Gd),np.transpose(Cl)),np.linalg.inv(B))  # Usado na Equação (15)

for m in range (time):
    
    teta[:,m]=np.dot(np.linalg.inv(B),np.transpose(P[m,:])) # Ângulo de tensão (Teta). Equação (14)
    grau[:,m]=np.dot(np.transpose(Cl),teta[:,m])            # Diferença de ângulo de tensão (Teta ij)
    PL2[m]=np.dot(2*Gv,1-np.cos(grau[:,m]))                 # Perdas de potência usando a equação (13) - Perdas reais
    PT[m]=np.sum([P[m,:]])                                  # Potência total
    rLoss[m]=np.divide(PL2[m],PT[m])                        # Perdas de potência (%)


# =============================================================================
# Calculo da Matrix X (para os dados de treino)
# =============================================================================
X = np.column_stack((
    P[:, 0]**2,             # P1^2
    2 * P[:, 0] * P[:, 1],  # 2*P1*P2
    2 * P[:, 0] * P[:, 2],  # 2*P1*P3
    2 * P[:, 0] * P[:, 3],  # 2*P1*P4
    P[:, 1]**2,             # P2^2
    2 * P[:, 1] * P[:, 2],  # 2*P2*P3
    2 * P[:, 1] * P[:, 3],  # 2*P2*P4
    P[:, 2]**2,             # P3^2
    2 * P[:, 2] * P[:, 3],  # 2*P3*P4
    P[:, 3]**2              # P4^2
))

# =============================================================================
# Calculo do novo beta (usando a solução OLS)
# beta_novo = (X^T * X)^(-1) * X^T * y, onde y corresponde a PL2 (perdas reais)
# =============================================================================
# Adicionar ruído aos valores de PL2
PL2 = PL2 + np.random.normal(0, noiseFactor, PL2.shape)
beta_novo = np.linalg.inv(X.T @ X) @ X.T @ PL2

# =============================================================================
# Treino: calcular as perdas preditas para o conjunto de treino
# =============================================================================
PL2_train_pred = X @ beta_novo

# Calcular métricas de erro para o conjunto de treino
train_rmse = np.sqrt(mean_squared_error(PL2, PL2_train_pred))
train_mae = mean_absolute_error(PL2, PL2_train_pred)

print("Erro de Treino - RMSE: ", train_rmse)
print("Erro de Treino - MAE:  ", train_mae)

# =============================================================================
# Teste: construir a matriz X_test usando os mesmos termos de segundo grau
# =============================================================================
X_test = np.column_stack((
    Ptest[:, 0]**2,             # P1^2
    2 * Ptest[:, 0] * Ptest[:, 1],  # 2*P1*P2
    2 * Ptest[:, 0] * Ptest[:, 2],  # 2*P1*P3
    2 * Ptest[:, 0] * Ptest[:, 3],  # 2*P1*P4
    Ptest[:, 1]**2,             # P2^2
    2 * Ptest[:, 1] * Ptest[:, 2],  # 2*P2*P3
    2 * Ptest[:, 1] * Ptest[:, 3],  # 2*P2*P4
    Ptest[:, 2]**2,             # P3^2
    2 * Ptest[:, 2] * Ptest[:, 3],  # 2*P3*P4
    Ptest[:, 3]**2              # P4^2
))

# Predição para os dados de teste
PL2_test_new = X_test @ beta_novo

# =============================================================================
# Comparação entre o treino e o teste
# =============================================================================

# Plot da comparação para o conjunto de treino (valores reais vs. preditos)
plt.figure(figsize=(10,6))
plt.plot(PL2, label='Treino: Perdas Reais (PL2)', marker='o')
plt.plot(PL2_train_pred, label='Treino: Perdas Preditas', marker='x')
plt.xlabel("Time step")
plt.ylabel("Perdas de Potência")
plt.title("Comparação de Perdas - Treino")
plt.legend()
plt.grid(True)
plt.show()

# Plot dos resultados para os dados de teste (somente as predições, 
# pois os valores reais de teste podem não estar disponíveis)
plt.figure(figsize=(10,6))
plt.plot(PL2_test_new, label='Teste: Perdas Preditas', marker='s', color='orange')
plt.xlabel("Time step")
plt.ylabel("Perdas de Potência")
plt.title("Predição de Perdas para Dados de Teste")
plt.legend()
plt.grid(True)
plt.show()
