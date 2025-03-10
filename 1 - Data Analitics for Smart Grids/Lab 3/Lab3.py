###############################################################################################
# Laboratório 3 - Estimação de Estado                                                         #
#                                                                                             #
# Grupo 13                                                                                    #
#                                                                                             #
# Membros:                                                                                    #
#   Francisco Tavares (103402)                                                                #
#   Marta Valente (103574)                                                                    #
###############################################################################################

# NOTAS:
# consideramos uma matriz diagonal porque nao consideramos os erros entre variaveis.

# ============================================================================================================================================
# Importação de bibliotecas
# ============================================================================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ============================================================================================================================================
# Parâmetros Globais
# ============================================================================================================================================
networkFactor = 100    # Para alterar as características da rede (Y)
cosPhi = 0.95          # Valor de teta
m = 100                # Número de Iterações   
sig = 0.5              # Fator de ruído 

# ============================================================================================================================================
# Funções para carregar e processar dados
# ============================================================================================================================================
def load_data():
    """Carrega os dados do arquivo Excel e retorna as informações necessárias."""
    Info = np.array(pd.read_excel(r'DASG_Prob2_new.xlsx', sheet_name='Info', header=None))
    
    # Informação sobre o barramento de referência
    SlackBus = Info[0, 1]
    
    # Informação da rede
    Net_Info = np.array(pd.read_excel(r'DASG_Prob2_new.xlsx', sheet_name='Y_Data'))
    
    # Informação de potência (treino)
    Power_Info = np.array(pd.read_excel(r'DASG_Prob2_new.xlsx', sheet_name='Load(t,Bus)'))
    Power_Info = np.delete(Power_Info, [0], 1)
    
    return SlackBus, Net_Info, Power_Info

def prepare_network(Net_Info, Power_Info, SlackBus):
    """Prepara as matrizes da rede e define os parâmetros iniciais."""
    # Calcular os valores de potência
    time = Power_Info.shape[0]
    P = np.dot(-Power_Info, np.exp(complex(0, 1) * np.arccos(cosPhi)))
    I = np.conj(P[2, :])
    
    # Determinar o número de barramentos
    nBus = max(np.max(Net_Info[:, 0]), np.max(Net_Info[:, 1]))
    
    # Criar a variável número de linhas e a matriz de admitância (Y)
    nLines = Net_Info.shape[0]
    Y = np.zeros((nBus, nBus), dtype=complex)
    
    # Completar a matriz Y e atualizar o número de linhas
    for i in range(Net_Info.shape[0]):
        y_aux = Net_Info[i, 2].replace(",", ".")
        y_aux = y_aux.replace("i", "j")
        Y[Net_Info[i, 0]-1, Net_Info[i, 0]-1] = Y[Net_Info[i, 0]-1, Net_Info[i, 0]-1] + complex(y_aux) * networkFactor
        Y[Net_Info[i, 1]-1, Net_Info[i, 1]-1] = Y[Net_Info[i, 1]-1, Net_Info[i, 1]-1] + complex(y_aux) * networkFactor
        Y[Net_Info[i, 0]-1, Net_Info[i, 1]-1] = Y[Net_Info[i, 0]-1, Net_Info[i, 1]-1] - complex(y_aux) * networkFactor
        Y[Net_Info[i, 1]-1, Net_Info[i, 0]-1] = Y[Net_Info[i, 1]-1, Net_Info[i, 0]-1] - complex(y_aux) * networkFactor

    # Remover o barramento de referência da matriz de admitância            
    Yl = np.delete(Y, np.s_[SlackBus-1], axis=0)
    Yl = np.delete(Yl, np.s_[SlackBus-1], axis=1)
    
    # Para todos os valores que não pertecem à diagonal de Y o sinal tem que ser trocado
    for i in range(nBus):
        for j in range(nBus):
            if i != j:  # Se não for um elemento da diagonal
                Y[i, j] = -Y[i, j]  # Inverter o sinal


    # Matriz de Condutância
    G = Yl.real
    
    # Matriz de Susceptância
    B = Yl.imag
    
    return Y, Yl, G, B, I, nBus, nLines


# ============================================================================================================================================
# Funções para estimação de estado
# ============================================================================================================================================
def state_estimation_1(Y, Yl, I):
    """
    1ª Estimação de Estado - considerando informação completa sobre correntes I12 and I54 
    (amplitude and angle) and not estimating V3
    
    Esta estimação usa apenas medidas em barramentos específicos e não estima V3.
    """
    print("\n=== 1ª Estimação de Estado ===\n")
    print("Considerando informação completa sobre correntes I12 e I54 (amplitude e ângulo) e não estimando V3")
    
    # Criação da Matriz
    b0s = np.zeros((4), dtype=np.complex128)
    A0s = np.zeros((4, 4), dtype=np.complex128)
    v0s = np.zeros((5), dtype=np.complex128)
    
    # Cálculo da Tensão (Referência)
    v0s[0:4] = 1 + np.dot(np.linalg.inv(Yl), I)
    v0s[4] = 1
    
    # Valores de medição z 
    b0s[0] = np.dot(Y[0, 1], (v0s[0] - v0s[1]))
    b0s[1] = np.dot(Y[3, 4], (1 - v0s[3]))
    b0s[2] = v0s[1]
    b0s[3] = 1
    
    # Matriz Hx
    A0s[0, 0] = np.dot(Y[0, 1], 1)
    A0s[0, 1] = np.dot(-Y[0, 1], 1)
    A0s[1, 2] = np.dot(-Y[3, 4], 1)
    A0s[1, 3] = np.dot(Y[3, 4], 1)
    A0s[2, 1] = 1
    A0s[3, 3] = 1
    
    # Variáveis de Estado (x) - Estas variáveis são as tensões estimadas (V1; V2; V4; V5)
    # x = (Hx.T * Hx)^-1 * Hx.T * z
    x0s = np.dot(np.dot(np.linalg.inv(np.dot(A0s.conj().T, A0s)), A0s.conj().T), b0s)

    # Mostrar resultados
    print("\nResultados da 1ª Estimação de Estado:")
    print("Tensão no barramento 1:", x0s[0])
    print("Tensão no barramento 2:", x0s[1])
    print("Tensão no barramento 4:", x0s[2])
    print("Tensão no barramento 5:", x0s[3])
    print("\nNOTA: V3 não é estimado neste método.")
    
    return x0s, v0s

def state_estimation_2a(Y, Yl, I):
    """
    2ª Estimação de Estado - considerando informação RMS das correntes
    sem considerar o peso das pseudo-medições (Matriz W)
    
    Esta estimação inclui a tensão V3 na estimação de estado através de pseudo-medições.
    """
    print("\n=== 2ª Estimação de Estado (a) ===\n")
    print("Sem considerar o peso das pseudo-medições (Matriz W)")
    print("Esta estimação INCLUI o cálculo da tensão V3")
    
    # Criação das Matrizes
    b0 = np.zeros((8), dtype=np.complex128)
    A = np.zeros((8, 5), dtype=np.complex128)
    v = np.zeros((5), dtype=np.complex128)
    
    # Cálculo da Tensão (Referência)
    v[0:4] = 1 + np.dot(np.linalg.inv(Yl), I)
    v[4] = 1
    
    # Valores de medição z (Neste caso, estamos a calcular as correntes e tensões, mas também poderíamos medi-las)
    b0[0] = np.dot(np.absolute(np.dot(-Y[0,1],(v[0]-v[1]))),np.exp(complex(0,-1)*np.arccos(cosPhi)))
    b0[1] = np.dot(np.absolute(np.dot(-Y[3,4],(1-v[3]))),np.exp(complex(0,-1)*np.arccos(cosPhi)))
    b0[2] = v[1]
    b0[3] = v[4]
    b0[4:8] = I  # Aqui adicionamos pseudo-medições baseadas na corrente I

    print(b0)

    # Matriz Hx 
    A[0, 0] = np.dot(Y[0, 1], 1)                      #  Y12
    A[0, 1] = np.dot(-Y[0, 1], 1)                     # -Y12
    A[1, 3] = np.dot(-Y[3, 4], 1)                     # -Y45
    A[1, 4] = np.dot(Y[3, 4], 1)                      #  Y45
    A[2, 1] = 1                                       #  1 
    A[3, 4] = 1                                       #  1
    # Estas próximas linhas são cruciais para a estimação de V3:
    A[4, 0] = np.dot(Y[0, 1] + Y[0, 2], 1)            #  Y12+Y13
    A[4, 1] = A[0, 1]                                 # -Y12
    A[4, 2] = np.dot(-Y[0, 2], 1)                     # -Y13 (relaciona V3 com V1)
    A[5, 0] = A[0, 1]                                 # -Y12
    A[5, 1] = np.dot(Y[0, 1] + Y[1,2], 1)             #  Y12+Y23
    A[5, 2] = np.dot(-Y[1, 2], 1)                     # -Y23 (relaciona V3 com V2)
    A[6, 0] = A[4, 2]                                 # -Y13
    A[6, 1] = A[5, 2]                                 # -Y23
    A[6, 2] = np.dot(Y[0, 2] + Y[1, 2] + Y[2, 3], 1)  #  Y13+Y23+Y34 (conexões diretas para V3)
    A[6, 3] = np.dot(-Y[2, 3], 1)                     # -Y34
    A[7, 2] = A[6, 3]                                 # -Y34 (relaciona V3 com V4)
    A[7, 3] = np.dot(Y[2, 3] + Y[3, 4], 1)            #  Y34+Y45
    A[7, 4] = A[1, 3]                                 # -Y45


    print(A)
    # Estimação sem consideração do ruído 
    x0 = np.dot(np.dot(np.linalg.inv(np.dot(A.conj().T, A)), A.conj().T), b0)
    print(x0)

    # Ruído a ser adicionado às pseudo-medições
    np.random.seed(42)  # Para reprodutibilidade
    e = np.random.normal(0.0, 1.0, size=(4, m)) * sig
    
    # Criação da Matriz 
    sx = np.zeros(5, dtype=np.complex128)  # CORREÇÃO: Usar complex128 para evitar erros de tipo
    rms = np.zeros((5, m))
    ei12a = np.zeros(m)
    ei54a = np.zeros(m)
    
    # Para cada vez que estimamos o estado, temos de estimar a corrente: I=(V1-V2)y

    for i in range(m):
        # Introduzir erro nas medições (Matriz z)
        b1 = np.zeros(len(b0), dtype=np.complex128)  # IMPORTANTE: Usar o mesmo tipo para b1
        b1[0:4] = b0[0:4]  # Mantém os valores originais para as primeiras 4 medidas
        b1[4:8] = I + e[:, i]
        
        # Estimar as tensões com base nas medições com erros
        x = np.dot(np.dot(np.linalg.inv(np.dot(A.conj().T, A)), A.conj().T), b1)
    
        # Valor acumulado das estimações
        sx = sx + x
    
        # Erros nas tensões
        rms[:, i] = np.sqrt(np.abs(np.dot((x - x0), np.conjugate(x - x0))))
    
        # Erros relativos da corrente (Para serem utilizados nos gráficos)
        ei12a[i] = np.divide(
            np.absolute(np.absolute(np.dot(Y[0, 1], (x[0] - x[1]))) - np.absolute(np.dot(Y[0, 1], (v[0] - v[1])))), np.absolute(np.dot(Y[0, 1], (v[0] - v[1])))
        )
        ei54a[i] = np.divide(
            np.absolute(np.absolute(np.dot(Y[3, 4], (1 - x[3]))) - np.absolute(np.dot(Y[3, 4], (1 - v[3])))), np.absolute(np.dot(Y[3, 4], (1 - v[3])))
        )
    
    # Estimação média da tensão
    x_avg = sx / m
    
    # Erro RMS médio da tensão
    ee = np.sum(rms, axis=1) / m
    
    # Mostrar resultados
    print("\nResultados da 2ª Estimação de Estado (sem pesos):")
    print("Tensão média estimada no barramento 1:", x_avg[0])
    print("Tensão média estimada no barramento 2:", x_avg[1])
    print("Tensão média estimada no barramento 3:", x_avg[2])  # V3 está sendo estimado!
    print("Tensão média estimada no barramento 4:", x_avg[3])
    print("Tensão média estimada no barramento 5:", x_avg[4])
    print("\nErro RMS médio da tensão:", ee)
    
    return x0, x_avg, ei12a, ei54a, v, A, b1, e

def state_estimation_2b(Y, A, b1, I, e, v):
    """
    2ª Estimação de Estado - considerando informação RMS das correntes
    considerando o peso das pseudo-medições (Matriz W)
    
    Esta estimação inclui a tensão V3 na estimação de estado e usa uma matriz
    de pesos W para dar mais confiança às medições reais e menos às pseudo-medições.
    """
    print("\n=== 2ª Estimação de Estado (b) ===\n")
    print("Considerando o peso das pseudo-medições (Matriz W)")
    print("Esta estimação INCLUI o cálculo da tensão V3")
    
    # Criação da Matriz
    b2 = np.zeros((8), dtype=np.complex128)
    b2 = np.copy(b1)

    # Matriz dos Pesos (W) 
    W = np.zeros((len(b1), len(b1)))
    
    # Medições reais (primeiros 4 elementos) recebem peso alto = maior confiança
    np.fill_diagonal(W[0:4], 0.001**-2)
    np.fill_diagonal(W[4:8,4:8], sig**-2)   
    
    # Estimação sem considerar o ruído, mas considerando o peso 
    xw0 = np.dot(np.dot(np.linalg.inv(np.dot(np.dot(A.conjugate().T, W), A)), np.dot(A.conjugate().T, W)), b2)


    # Criação da Matriz
    sx = np.zeros(5, dtype=np.complex128)  # CORREÇÃO: Usar complex128 para evitar erros de tipo
    rms = np.zeros((5, m))
    ei12a = np.zeros(m)
    ei54a = np.zeros(m)
    
    for i in range(m):
        # Reset to original measurements each iteration
        b2 = np.copy(b1)
        
        # Add noise only to pseudo-measurements
        b2[4:8] = I + e[:, i]
    
        # Estimate voltages using complete measurement vector
        xw = np.dot(np.dot(np.linalg.inv(np.dot(np.dot(A.conjugate().T, W), A)), np.dot(A.conjugate().T, W)), b2)
        
        # Valor acumulado das estimações
        sx = sx + xw
    
        # Erros nas tensões
        rms[:, i] = np.sqrt(np.abs(np.dot((xw - xw0), np.conjugate(xw - xw0))))
    
        # Erros relativos da corrente (Para serem utilizados nos gráficos)
        ei12a[i] = np.divide(
            np.absolute(np.absolute(np.dot(Y[0, 1], (xw[0] - xw[1]))) - np.absolute(np.dot(Y[0, 1], (v[0] - v[1])))),
            np.absolute(np.dot(Y[0, 1], (v[0] - v[1])))
        )
        ei54a[i] = np.divide(
            np.absolute(np.absolute(np.dot(Y[3, 4], (1 - xw[3]))) - np.absolute(np.dot(Y[3, 4], (1 - v[3])))),
            np.absolute(np.dot(Y[3, 4], (1 - v[3])))
        )
            
    # Estimação média da tensão
    xw_avg = sx / m
    
    # Erro RMS médio da tensão
    eew = np.sum(rms, axis=1) / m
    
    # Mostrar resultados
    print("\nResultados da 2ª Estimação de Estado (com pesos):")
    print("Tensão média estimada no barramento 1:", xw_avg[0])
    print("Tensão média estimada no barramento 2:", xw_avg[1])
    print("Tensão média estimada no barramento 3:", xw_avg[2])
    print("Tensão média estimada no barramento 4:", xw_avg[3])
    print("Tensão média estimada no barramento 5:", xw_avg[4])
    print("\nErro RMS médio da tensão:", eew)
    
    return xw0, xw_avg, ei12a, ei54a


# ============================================================================================================================================
# Funções para o Desafio 1 - selecionando apenas duas pseudo-medições
# ============================================================================================================================================
def state_estimation_alternative(Y, Yl, I, pseudo_selection=[0, 1]):
    """
    Estimação de Estado Alternativa - Utilizando apenas 2 pseudo-medições das 4 disponíveis.
    
    Args:
        Y: Matriz de admitância completa
        Yl: Matriz de admitância sem barramento de referência
        I: Vetor de corrente
        pseudo_selection: Lista com índices das duas pseudo-medições a serem utilizadas [0-3]
    
    Returns:
        Resultados da estimação e erros associados
    """
    print("\n=== Estimação de Estado Alternativa ===\n")
    print(f"Utilizando apenas as pseudo-medições {[i+1 for i in pseudo_selection]}")
    
    # Criar matrizes base como na estimação 2a
    b0 = np.zeros((8), dtype=np.complex128)
    A = np.zeros((8, 5), dtype=np.complex128)
    v = np.zeros((5), dtype=np.complex128)
    
    # Cálculo da Tensão (Referência)
    v[0:4] = 1 + np.dot(np.linalg.inv(Yl), I)
    v[4] = 1
    
    # Valores de medição z 
    b0[0] = np.absolute(np.dot(Y[0, 1], (v[0] - v[1]))) * np.exp(complex(0, -1) * np.arccos(cosPhi))
    b0[1] = np.absolute(np.dot(Y[3, 4], (v[4] - v[3]))) * np.exp(complex(0, -1) * np.arccos(cosPhi))
    b0[2] = v[1]
    b0[3] = v[4]
    b0[4:8] = I  # Todas as pseudo-medições baseadas na corrente I
    
    # Matriz Hx 
    A[0, 0] = np.dot(Y[0, 1], 1)                      #  Y12
    A[0, 1] = np.dot(-Y[0, 1], 1)                     # -Y12
    A[1, 3] = np.dot(-Y[3, 4], 1)                     # -Y45
    A[1, 4] = np.dot(Y[3, 4], 1)                      #  Y45
    A[2, 1] = 1                                       #  1 
    A[3, 4] = 1                                       #  1
    # Estas próximas linhas são cruciais para a estimação de V3:
    A[4, 0] = np.dot(Y[0, 1] + Y[0, 2], 1)            #  Y12+Y13
    A[4, 1] = A[0, 1]                                 # -Y12
    A[4, 2] = np.dot(-Y[0, 2], 1)                     # -Y13 (relaciona V3 com V1)
    A[5, 0] = A[0, 1]                                 # -Y12
    A[5, 1] = np.dot(Y[0, 1] + Y[1,2], 1)             #  Y12+Y23
    A[5, 2] = np.dot(-Y[1, 2], 1)                     # -Y23 (relaciona V3 com V2)
    A[6, 0] = A[4, 2]                                 # -Y13
    A[6, 1] = A[5, 2]                                 # -Y23
    A[6, 2] = np.dot(Y[0, 2] + Y[1, 2] + Y[2, 3], 1)  #  Y13+Y23+Y34 (conexões diretas para V3)
    A[6, 3] = np.dot(-Y[2, 3], 1)                     # -Y34
    A[7, 2] = A[6, 3]                                 # -Y34 (relaciona V3 com V4)
    A[7, 3] = np.dot(Y[2, 3] + Y[3, 4], 1)            #  Y34+Y45
    A[7, 4] = A[1, 3]                                 # -Y45
    
    # Manter as 4 medições reais (índices 0-3) e adicionar as 2 pseudo-medições selecionadas
    selected_indices = list(range(4)) + [4 + i for i in pseudo_selection]
    
    b0_selected = b0[selected_indices]
    A_selected = A[selected_indices, :]
    
    # Matriz de pesos
    W = np.zeros((len(b0_selected), len(b0_selected)))
    np.fill_diagonal(W[0:4, 0:4], 10**6)  # Alto peso para medições reais
    np.fill_diagonal(W[4:, 4:], 2)        # Peso para todas as pseudo-medições selecionadas
    
    # Ruído a ser adicionado às pseudo-medições
    np.random.seed(42)
    e_indices = [i for i in pseudo_selection]
    e = np.random.normal(0.0, 1.0, size=(len(e_indices), m)) * sig
    
    # Estimação sem ruído
    x0 = np.dot(np.dot(np.linalg.inv(np.dot(np.dot(A_selected.conj().T, W), A_selected)), 
                np.dot(A_selected.conj().T, W)), b0_selected)
    
    # Variáveis para acumular resultados
    sx = np.zeros(5, dtype=np.complex128)
    rms = np.zeros((5, m))
    ei12a = np.zeros(m)
    ei54a = np.zeros(m)
    
    for i in range(m):
        # Introduzir erro apenas nas pseudo-medições selecionadas
        b1 = np.copy(b0_selected)
        for idx, e_idx in enumerate(e_indices):
            b1[4 + idx] = I[e_idx] + e[idx, i]
        
        
        # Estimar tensões com erros e pesos
        x = np.dot(np.dot(np.linalg.inv(np.dot(np.dot(A_selected.conj().T, W), A_selected)), 
                   np.dot(A_selected.conj().T, W)), b1)
        
        # Acumular estimações
        sx = sx + x
        
        # Erros nas tensões
        rms[:, i] = np.sqrt(np.abs(np.dot((x - x0), np.conjugate(x - x0))))
        
        # Erros relativos das correntes
        ei12a[i] = np.divide(
            np.absolute(np.dot(Y[0, 1], (x[0] - x[1]))) - np.absolute(np.dot(Y[0, 1], (v[0] - v[1]))),
            np.absolute(np.dot(Y[0, 1], (v[0] - v[1])))
        )
        ei54a[i] = np.divide(
            np.absolute(np.dot(Y[3, 4], (x[4] - x[3]))) - np.absolute(np.dot(Y[3, 4], (v[4] - v[3]))),
            np.absolute(np.dot(Y[3, 4], (v[4] - v[3])))
        )
    
    # Estimação média
    x_avg = sx / m
    
    # Erro RMS médio
    ee = np.sum(rms, axis=1) / m
    
    # Mostrar resultados
    print("\nResultados da Estimação Alternativa:")
    print("Tensão média estimada no barramento 1:", x_avg[0])
    print("Tensão média estimada no barramento 2:", x_avg[1])
    print("Tensão média estimada no barramento 3:", x_avg[2])
    print("Tensão média estimada no barramento 4:", x_avg[3])
    print("Tensão média estimada no barramento 5:", x_avg[4])
    print("\nErro RMS médio da tensão:", ee)
    print("Erro RMS médio total:", np.sum(ee))
    
    return x0, x_avg, ei12a, ei54a, ee, A_selected, b0_selected, e

def evaluate_all_combinations(Y, Yl, I):
    """
    Avalia todas as combinações possíveis de 2 pseudo-medições usando o traço da matriz de covariância.
    """
    import itertools
    
    print("\n=== Avaliação de Todas as Combinações de Pseudo-medições (Versão Melhorada) ===\n")
    
    combinations = list(itertools.combinations(range(4), 2))
    results = {}
    best_trace = float('inf')
    best_combo = None
    
    # Criar matrizes base como na estimação 2a
    b0 = np.zeros((8), dtype=np.complex128)
    A = np.zeros((8, 5), dtype=np.complex128)
    v = np.zeros((5), dtype=np.complex128)
    
    # Cálculo da Tensão (Referência)
    v[0:4] = 1 + np.dot(np.linalg.inv(Yl), I)
    v[4] = 1
    
    # Valores de medição z 
    b0[0] = np.absolute(np.dot(Y[0, 1], (v[0] - v[1]))) * np.exp(complex(0, -1) * np.arccos(cosPhi))
    b0[1] = np.absolute(np.dot(Y[3, 4], (v[4] - v[3]))) * np.exp(complex(0, -1) * np.arccos(cosPhi))
    b0[2] = v[1]
    b0[3] = v[4]
    b0[4:8] = I

    # Matriz Hx 
    A[0, 0] = np.dot(Y[0, 1], 1)                      #  Y12
    A[0, 1] = np.dot(-Y[0, 1], 1)                     # -Y12
    A[1, 3] = np.dot(-Y[3, 4], 1)                     # -Y45
    A[1, 4] = np.dot(Y[3, 4], 1)                      #  Y45
    A[2, 1] = 1                                       #  1 
    A[3, 4] = 1                                       #  1
    # Estas próximas linhas são cruciais para a estimação de V3:
    A[4, 0] = np.dot(Y[0, 1] + Y[0, 2], 1)            #  Y12+Y13
    A[4, 1] = A[0, 1]                                 # -Y12
    A[4, 2] = np.dot(-Y[0, 2], 1)                     # -Y13 (relaciona V3 com V1)
    A[5, 0] = A[0, 1]                                 # -Y12
    A[5, 1] = np.dot(Y[0, 1] + Y[1,2], 1)             #  Y12+Y23
    A[5, 2] = np.dot(-Y[1, 2], 1)                     # -Y23 (relaciona V3 com V2)
    A[6, 0] = A[4, 2]                                 # -Y13
    A[6, 1] = A[5, 2]                                 # -Y23
    A[6, 2] = np.dot(Y[0, 2] + Y[1, 2] + Y[2, 3], 1)  #  Y13+Y23+Y34 (conexões diretas para V3)
    A[6, 3] = np.dot(-Y[2, 3], 1)                     # -Y34
    A[7, 2] = A[6, 3]                                 # -Y34 (relaciona V3 com V4)
    A[7, 3] = np.dot(Y[2, 3] + Y[3, 4], 1)            #  Y34+Y45
    A[7, 4] = A[1, 3]                                 # -Y45

    # Avaliar cada combinação
    for combo in combinations:
        # Selecionar índices
        selected_indices = list(range(4)) + [4 + i for i in combo]
        A_selected = A[selected_indices, :]
        
        # Matriz de pesos
        W = np.zeros((len(selected_indices), len(selected_indices)))
        np.fill_diagonal(W[0:4, 0:4], 10**6)  # Alto peso para medições reais
        np.fill_diagonal(W[4:, 4:], 2)        # Peso para as pseudo-medições selecionadas
        
        # Calcular matriz de covariância G^-1
        G_inv = np.linalg.inv(np.dot(np.dot(A_selected.conj().T, W), A_selected))
        
        # Calcular o traço (soma dos elementos da diagonal)
        trace = np.abs(np.trace(G_inv))
        
        results[combo] = {
            'trace': trace
        }
        
        print(f"Pseudo-medições {[i+1 for i in combo]} - Traço: {trace:.6f}")
        
        # Verificar se é o melhor até agora
        if trace < best_trace:
            best_trace = trace
            best_combo = combo
    
    print(f"\nMelhor combinação baseada no traço da matriz G^-1: {[i+1 for i in best_combo]}")
    print(f"Valor do traço: {best_trace:.6f}")
    
    # Visualizar resultados
    plot_trace_comparison(results)
    
    return results, best_combo

def plot_trace_comparison(results):
    """Plota comparação do traço da matriz G^-1 para todas as combinações."""
    combos = list(results.keys())
    traces = [results[combo]['trace'] for combo in combos]
    combo_labels = [f"S{c[0]+1}+S{c[1]+1}" for c in combos]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(combo_labels, traces)
    
    # Destaque para o melhor resultado
    best_idx = np.argmin(traces)
    bars[best_idx].set_color('green')
    
    plt.xlabel('Combinações de Pseudo-medições')
    plt.ylabel('Traço da Matriz G^-1')
    plt.title('Comparação do Traço da Matriz de Covariância')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adicionar valores sobre as barras
    for i, v in enumerate(traces):
        plt.text(i, v + 0.01, f"{v:.5f}", ha='center', fontsize=9)
    
    plt.savefig('figuras/comparacao_trace_combinacoes.png', dpi=300)
    plt.tight_layout()
    plt.show()

def compare_with_full_approach(Y, Yl, I, best_combo, results_2b):
    """
    Compara a abordagem de 2 pseudo-medições com a abordagem completa (4 pseudo-medições).
    
    Args:
        Y: Matriz de admitância completa
        Yl: Matriz de admitância sem barramento de referência
        I: Vetor de corrente
        best_combo: Melhor combinação de pseudo-medições
        results_2b: Resultados da abordagem completa (state_estimation_2b)
    """
    print("\n=== Comparação com Abordagem Completa ===\n")
    
    # Executar estimação com a melhor combinação
    _, x_avg_alt, ei12a_alt, ei54a_alt, ee_alt, _, _, _ = state_estimation_alternative(Y, Yl, I, best_combo)
    
    # Obter resultados da abordagem completa
    _, xw_avg_2b, _, _ = results_2b
    
    # Calcular diferenças percentuais nas magnitudes
    diff_mag = []
    for i in range(5):
        mag_alt = np.abs(x_avg_alt[i])
        mag_2b = np.abs(xw_avg_2b[i])
        diff_pct = abs(mag_alt - mag_2b) / mag_2b * 100
        diff_mag.append(diff_pct)
    
    # Visualizar comparação
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Gráfico de barras para magnitudes de tensão
    bus_numbers = [1, 2, 3, 4, 5]
    width = 0.35
    
    ax1.bar([x - width/2 for x in bus_numbers], np.abs(x_avg_alt), width, 
           label=f'2 Pseudo-medições (S{best_combo[0]+1}+S{best_combo[1]+1})', alpha=0.7)
    ax1.bar([x + width/2 for x in bus_numbers], np.abs(xw_avg_2b), width, 
           label='4 Pseudo-medições (Completa)', alpha=0.7)
    
    ax1.set_xlabel('Número do Barramento')
    ax1.set_ylabel('Magnitude da Tensão')
    ax1.set_title('Comparação das Magnitudes de Tensão')
    ax1.set_xticks(bus_numbers)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Gráfico de barras para diferenças percentuais
    ax2.bar(bus_numbers, diff_mag, color='orange', alpha=0.7)
    ax2.set_xlabel('Número do Barramento')
    ax2.set_ylabel('Diferença Percentual (%)')
    ax2.set_title('Diferença Percentual nas Magnitudes')
    ax2.set_xticks(bus_numbers)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Adicionar valores sobre as barras
    for i, v in enumerate(diff_mag):
        ax2.text(i+1, v + 0.05, f"{v:.2f}%", ha='center')
    
    plt.tight_layout()
    
    # Salvar figura
    if not os.path.exists('figuras'):
        os.makedirs('figuras')
    plt.savefig('figuras/comparacao_abordagens.png', dpi=300)
    
    plt.show()
    
    print("\nDiferença percentual nas magnitudes de tensão:")
    for i, bus in enumerate(bus_numbers):
        print(f"Barramento {bus}: {diff_mag[i]:.2f}%")


# ============================================================================================================================================
# Funções para o Desafio 2 - I12=0
# ============================================================================================================================================
def analyze_scenario_I12_zero(Y, I, cosPhi):
    """
    Analyze which scenario is more likely when I12=0:
    1) Branch 1-2 is out of service (y12=0)
    2) The current measurement device for I12 is malfunctioning
    
    The analysis compares residuals from least squares estimation for both scenarios
    and determines which one has the smaller error.
    
    Args:
        Y: Admittance matrix (complex)
        I: Current injection vector (complex)
        cosPhi: Power factor
        
    Returns:
        Dictionary with results for both scenarios and conclusion
    """
    # -------------------------
    # Calculate reference voltages
    # -------------------------
    Yl = np.delete(Y, 0, axis=0)
    Yl = np.delete(Yl, 0, axis=1)
    v = np.zeros(5, dtype=complex)
    v[0:4] = 1 + np.dot(np.linalg.inv(Yl), I)
    v[4] = 1

    # -------------------------
    # Build measurement vector (z) and sensitivity matrix (H)
    # -------------------------
    # Measurement vector (8 elements):
    # z[0]: I12 current flow between buses 1-2
    # z[1]: I54 current flow magnitude with phase shift
    # z[2]: V2 voltage at bus 2
    # z[3]: V5 voltage at bus 5
    # z[4:8]: pseudo-measurements (injection currents)
    b0 = np.zeros(8, dtype=complex)
    b0[0] = np.dot(Y[0, 1], (v[0] - v[1]))
    b0[1] = np.absolute(np.dot(Y[3, 4], (v[4] - v[3]))) * np.exp(complex(0, -1) * np.arccos(cosPhi))
    b0[2] = v[1]
    b0[3] = v[4]
    b0[4:8] = I  # pseudo-measurements

    # Sensitivity matrix H (8x5) - defining relationships per model
    A = np.zeros((8, 5), dtype=complex)
    A[0, 0] = Y[0, 1]
    A[0, 1] = -Y[0, 1]
    A[1, 3] = -Y[3, 4]
    A[1, 4] = Y[3, 4]
    A[2, 1] = 1
    A[3, 4] = 1
    # Relationships to include V3:
    A[4, 0] = Y[0, 1] + Y[0, 2]
    A[4, 1] = -Y[0, 1]
    A[4, 2] = -Y[0, 2]
    A[5, 0] = -Y[0, 1]
    A[5, 1] = Y[0, 1] + Y[1, 2]
    A[5, 2] = -Y[1, 2]
    A[6, 0] = -Y[0, 2]
    A[6, 1] = -Y[1, 2]
    A[6, 2] = Y[0, 2] + Y[1, 2] + Y[2, 3]
    A[6, 3] = -Y[2, 3]
    A[7, 2] = -Y[2, 3]
    A[7, 3] = Y[2, 3] + Y[3, 4]
    A[7, 4] = -Y[3, 4]

    # -------------------------
    # Scenario 1: Branch 1-2 out of service
    # In this case, we remove I12 measurement and corresponding row from H matrix
    # -------------------------
    print("\n--- Scenario 1: Branch 1-2 out of service ---")
    b_branch = np.delete(b0, 0, axis=0)      # remove I12 measurement
    A_branch = np.delete(A, 0, axis=0)       # remove corresponding row in H
    
    # State estimation using pseudoinverse (more stable for potentially ill-conditioned matrices)
    # Using conjugate transpose (Hermitian) for complex matrices: A^H·A
    x_branch = np.dot(np.linalg.pinv(A_branch), b_branch)
    
    # Calculate residuals and error norm
    residue_branch = b_branch - np.dot(A_branch, x_branch)
    error_branch = np.linalg.norm(residue_branch)
    print(f"Estimated voltages: {np.abs(x_branch)}")
    print(f"Residual norm: {error_branch:.6f}")

    # -------------------------
    # Scenario 2: I12 meter malfunction
    # We keep full structure but set I12 measurement to zero
    # -------------------------
    print("\n--- Scenario 2: I12 meter malfunction ---")
    b_meter = b0.copy()
    b_meter[0] = 0  # Force I12 measurement to zero
    
    # State estimation with full structure
    x_meter = np.dot(np.linalg.pinv(A), b_meter)
    
    # Calculate residuals and error norm
    residue_meter = b_meter - np.dot(A, x_meter)
    error_meter = np.linalg.norm(residue_meter)
    print(f"Estimated voltages: {np.abs(x_meter)}")
    print(f"Residual norm: {error_meter:.6f}")

    # -------------------------
    # Compare scenarios and determine most likely one
    # -------------------------
    print("\n--- Comparison of Scenarios ---")
    print(f"Error (residual norm) - Branch out of service: {error_branch:.6f}")
    print(f"Error (residual norm) - Meter malfunction: {error_meter:.6f}")
    
    # Confidence metric: relative difference between errors
    # Note: This is a simple heuristic and not a formal statistical test
    if error_branch < error_meter:
        confidence = 100 * (1 - error_branch / error_meter)
        conclusion = f"Branch 1-2 is most likely out of service (y12=0) with {confidence:.2f}% confidence"
    else:
        confidence = 100 * (1 - error_meter / error_branch)
        conclusion = f"I12 meter is most likely malfunctioning with {confidence:.2f}% confidence"
    
    print(f"Conclusion: {conclusion}")
    
    return {
        "branch_out": {
            "x": x_branch,
            "residue": residue_branch,
            "error": error_branch
        },
        "meter_fault": {
            "x": x_meter,
            "residue": residue_meter,
            "error": error_meter
        },
        "conclusion": conclusion
    }

def analyze_scenario_I12_zero_weighted(Y, I, cosPhi, weight_type='identity'):
    """
    Analyze which scenario is more likely when I12=0 using weighted least squares.
    
    Args:
        Y: Admittance matrix
        I: Current injection vector
        cosPhi: Power factor
        weight_type: Type of weight matrix to use
                    'identity': Equal weights for all measurements
                    'diagonal': Higher weights for real measurements
                    'custom': Custom weights based on measurement type
    """
    # Calculate reference voltages
    Yl = np.delete(Y, 0, axis=0)
    Yl = np.delete(Yl, 0, axis=1)
    v = np.zeros(5, dtype=complex)
    v[0:4] = 1 + np.dot(np.linalg.inv(Yl), I)
    v[4] = 1

    # Build measurement vector and sensitivity matrix
    b0 = np.zeros(8, dtype=complex)
    b0[0] = np.dot(Y[0, 1], (v[0] - v[1]))
    b0[1] = np.absolute(np.dot(Y[3, 4], (v[4] - v[3]))) * np.exp(complex(0, -1) * np.arccos(cosPhi))
    b0[2] = v[1]
    b0[3] = v[4]
    b0[4:8] = I  # pseudo-measurements

    # Sensitivity matrix
    A = np.zeros((8, 5), dtype=complex)
    # Fill in A as in your original function...
    
    # Create weight matrix W based on specified type
    W = np.zeros((8, 8))
    
    if weight_type == 'identity':
        # Equal weights for all measurements
        np.fill_diagonal(W, 1)
    elif weight_type == 'diagonal':
        # Higher weights for real measurements
        np.fill_diagonal(W[0:4, 0:4], 10)  # Real measurements
        np.fill_diagonal(W[4:8, 4:8], 1)   # Pseudo-measurements
    elif weight_type == 'custom':
        # Custom weights based on measurement type
        W[0, 0] = 5    # I12
        W[1, 1] = 8    # I54 
        W[2, 2] = 10   # V2
        W[3, 3] = 10   # V5
        np.fill_diagonal(W[4:8, 4:8], 0.5) # Pseudo-measurements
    
    # Scenario 1: Branch 1-2 out of service
    print(f"\n--- Scenario 1: Branch out (Weight: {weight_type}) ---")
    b_branch = np.delete(b0, 0, axis=0)
    A_branch = np.delete(A, 0, axis=0)
    W_branch = np.delete(np.delete(W, 0, axis=0), 0, axis=1)
    
    # Weighted least squares estimation:
    # x = (A^H W A)^-1 A^H W z
    AH_W_A = np.dot(np.dot(A_branch.conj().T, W_branch), A_branch)
    x_branch = np.dot(np.dot(np.linalg.inv(AH_W_A), np.dot(A_branch.conj().T, W_branch)), b_branch)
    
    # Calculate covariance matrix and residuals
    cov_branch = np.linalg.inv(AH_W_A)  # Rx = (A^H W A)^-1
    residue_branch = b_branch - np.dot(A_branch, x_branch)
    error_branch = np.linalg.norm(residue_branch)
    
    # Scenario 2: I12 meter malfunction
    # Similar implementation...

    # Compare scenarios
    # ...

    return {
        "branch_out": {
            "x": x_branch,
            "residue": residue_branch,
            "error": error_branch,
            "covariance": cov_branch
        },
        "meter_fault": {
            "x": x_meter,
            "residue": residue_meter,
            "error": error_meter,
            "covariance": cov_meter
        },
        "weight_type": weight_type,
        "conclusion": conclusion
    }

def plot_residue_comparison(residue_branch, residue_meter):
    """Plota um gráfico comparativo dos resíduos dos dois cenários."""
    abs_res_branch = np.abs(residue_branch)
    abs_res_meter = np.abs(residue_meter)
    
    plt.figure(figsize=(10, 6))
    indices_branch = np.arange(len(abs_res_branch))
    indices_meter = np.arange(len(abs_res_meter))
    
    width = 0.35
    plt.bar(indices_branch, abs_res_branch, width, label='Ramo fora de serviço', alpha=0.7)
    plt.bar(indices_meter + width, abs_res_meter, width, label='Medidor com falha', alpha=0.7)
    
    plt.xlabel('Índice da Medição')
    plt.ylabel('Magnitude do Resíduo')
    plt.title('Comparação dos Resíduos para os Cenários I12=0')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if not os.path.exists('figuras'):
        os.makedirs('figuras')
    plt.savefig('figuras/residuo_cenarios.png', dpi=300)
    plt.tight_layout()
    plt.show()

def compare_with_full_approach_challenge_2(Y, Yl, I, results_2b):
    """
    Compara a abordagem do Desafio 2 (I₁₂=0) com a abordagem completa (estimação normal).
    
    Args:
        Y: Matriz de admitância completa
        Yl: Matriz de admitância sem o barramento de referência
        I: Vetor de correntes nodais
        results_2b: Resultados da 2ª Estimação de Estado (com pesos)
    """
    print("\n=== Comparação do Desafio 2 com Abordagem Completa ===\n")
    
    # Get Challenge 2 results using the analysis function
    scenario_results = analyze_scenario_I12_zero(Y, I, cosPhi)
    
    # Determine which scenario (branch out or meter fault) had lower error
    if scenario_results['branch_out']['error'] < scenario_results['meter_fault']['error']:
        chosen_voltage = scenario_results['branch_out']['x']
        scenario_name = "Branch out of service"
    else:
        chosen_voltage = scenario_results['meter_fault']['x']
        scenario_name = "Meter malfunction"
    
    # Obter a estimativa completa (abordagem normal) dos resultados 2b
    xw_avg_2b = results_2b[1]
    
    # Calcular diferenças percentuais nas magnitudes
    diff_mag = []
    for i in range(5):
        mag_scenario = np.abs(chosen_voltage[i])
        mag_normal = np.abs(xw_avg_2b[i])
        diff_pct = abs(mag_scenario - mag_normal) / mag_normal * 100
        diff_mag.append(diff_pct)
    
    # Visualizar a comparação com gráficos
    bus_numbers = [1, 2, 3, 4, 5]
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Gráfico de barras para as magnitudes de tensão
    ax1.bar([x - width/2 for x in bus_numbers], [np.abs(chosen_voltage[i]) for i in range(5)], width, 
            label=f"Desafio 2 ({scenario_name})", alpha=0.7)
    ax1.bar([x + width/2 for x in bus_numbers], [np.abs(xw_avg_2b[i]) for i in range(5)], width, 
            label='Abordagem Completa', alpha=0.7)
    
    ax1.set_xlabel('Número do Barramento')
    ax1.set_ylabel('Magnitude da Tensão')
    ax1.set_title('Comparação das Magnitudes de Tensão')
    ax1.set_xticks(bus_numbers)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Gráfico de barras para as diferenças percentuais
    ax2.bar(bus_numbers, diff_mag, color='orange', alpha=0.7)
    ax2.set_xlabel('Número do Barramento')
    ax2.set_ylabel('Diferença Percentual (%)')
    ax2.set_title('Diferença Percentual nas Magnitudes')
    ax2.set_xticks(bus_numbers)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Adicionar valores sobre as barras
    for i, v in enumerate(diff_mag):
        ax2.text(i+1, v + 0.05, f"{v:.2f}%", ha='center')

    plt.tight_layout()

    # Salvar figura
    if not os.path.exists('figuras'):
        os.makedirs('figuras')
    plt.savefig('figuras/comparacao_desafio2_abordagem_completa.png', dpi=300)

    plt.show()

    print("\nDiferença percentual nas magnitudes de tensão:")
    for i, bus in enumerate(bus_numbers):
        print(f"Barramento {bus}: {diff_mag[i]:.2f}%")

    return diff_mag

def compare_weight_matrices_effect(Y, I, cosPhi):
    """Compare effects of different weight matrices on the scenario analysis."""
    weight_types = ['identity', 'diagonal', 'custom']
    results = {}
    
    for wtype in weight_types:
        results[wtype] = analyze_scenario_I12_zero_weighted(Y, I, cosPhi, wtype)
    
    # Create comparison plots
    fig, (def compare_weight_matrices_effect(Y, I, cosPhi):
    """Compare effects of different weight matrices on the scenario analysis."""
    weight_types = ['identity', 'diagonal', 'custom']
    results = {}
    
    for wtype in weight_types:
        results[wtype] = analyze_scenario_I12_zero_weighted(Y, I, cosPhi, wtype)
    
    # Create comparison plots
    fig, ( 

# ============================================================================================================================================
# Funções para visualização de resultados
# ============================================================================================================================================
def plot_errors_2a(ei12a, ei54a):
    """Plota os erros relativos para a estimação 2a."""
    plt.figure(figsize=(10, 5))
    plt.plot(ei12a, label='Erro Relativo da Corrente I12')
    plt.plot(ei54a, label='Erro Relativo da Corrente I54')
    plt.xlabel('Iteração')
    plt.ylabel('Erro Relativo')
    plt.title('Erro Relativo das Correntes (Sem Considerar Pesos)')
    plt.legend()
    plt.grid(True)
    
    # Salvar figura se necessário
    if not os.path.exists('figuras'):
        os.makedirs('figuras')
    plt.savefig('figuras/erro_relativo_sem_pesos.png', dpi=300)
    
    plt.show()

def plot_errors_2b(ei12a, ei54a):
    """Plota os erros relativos para a estimação 2b."""
    plt.figure(figsize=(10, 5))
    plt.plot(ei12a, label='Erro Relativo da Corrente I12')
    plt.plot(ei54a, label='Erro Relativo da Corrente I54')
    plt.xlabel('Iteração')
    plt.ylabel('Erro Relativo')
    plt.title('Erro Relativo das Correntes (Considerando Pesos)')
    plt.legend()
    plt.grid(True)
    
    # Salvar figura se necessário
    if not os.path.exists('figuras'):
        os.makedirs('figuras')
    plt.savefig('figuras/erro_relativo_com_pesos.png', dpi=300)
    
    plt.show()

def plot_error_scatter(ei12a_2a, ei54a_2a, ei12a_2b, ei54a_2b):
    """
    Cria um gráfico de dispersão que compara os erros relativos 
    das correntes I12 e I54 entre as duas abordagens (com e sem pesos).
    
    Args:
        ei12a_2a: Erro relativo de I12 sem pesos
        ei54a_2a: Erro relativo de I54 sem pesos
        ei12a_2b: Erro relativo de I12 com pesos
        ei54a_2b: Erro relativo de I54 com pesos
    """
    
    # Criar o gráfico de dispersão
    plt.figure(figsize=(10, 8))
    plt.scatter(ei12a_2a, ei54a_2a, label="Sem Pesos (W)", color="blue", alpha=0.7, marker='o')
    plt.scatter(ei12a_2b, ei54a_2b, label="Com Pesos (W)", color="orange", alpha=0.7, marker='x')
    
    # Configuração do gráfico
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Erro Relativo |I12|")
    plt.ylabel("Erro Relativo |I54|")
    plt.title("Comparação de Erros Relativos das Correntes")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    # Salvar figura se necessário
    if not os.path.exists('figuras'):
        os.makedirs('figuras')
    plt.savefig('figuras/erro_scatter_plot.png', dpi=300)
    
    plt.tight_layout()
    plt.show()

def compare_estimations(ei12a_2a, ei54a_2a, ei12a_2b, ei54a_2b):
    """Compara os resultados das estimações 2a e 2b."""
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(ei12a_2a, label='Sem Pesos', color='blue')
    plt.plot(ei12a_2b, label='Com Pesos', color='red')
    plt.title('Comparação do Erro Relativo da Corrente I12')
    plt.xlabel('Iteração')
    plt.ylabel('Erro Relativo')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(ei54a_2a, label='Sem Pesos', color='blue')
    plt.plot(ei54a_2b, label='Com Pesos', color='red')
    plt.title('Comparação do Erro Relativo da Corrente I54')
    plt.xlabel('Iteração')
    plt.ylabel('Erro Relativo')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Salvar figura se necessário
    if not os.path.exists('figuras'):
        os.makedirs('figuras')
    plt.savefig('figuras/comparacao_estimacoes.png', dpi=300)
    
    plt.show()

def compare_all_estimations(results):
    """Compara todos os resultados das estimações, incluindo onde V3 é estimado."""
    if results['est1'] is None or results['est2a'] is None or results['est2b'] is None:
        print("\nÉ necessário executar todas as estimações primeiro!")
        return
        
    x0s, v0s = results['est1']
    _, x_avg_2a, _, _, _, _, _, _ = results['est2a']
    _, xw_avg_2b, _, _ = results['est2b']
    
    print("\n=== Comparação de Todas as Estimações ===\n")
    print("Barramento 1:")
    print("  - Estimação 1:           ", x0s[0])
    print("  - Estimação 2a (sem W):  ", x_avg_2a[0])
    print("  - Estimação 2b (com W):  ", xw_avg_2b[0])
    
    print("\nBarramento 2:")
    print("  - Estimação 1:           ", x0s[1])
    print("  - Estimação 2a (sem W):  ", x_avg_2a[1])
    print("  - Estimação 2b (com W):  ", xw_avg_2b[1])
    
    print("\nBarramento 3:")
    print("  - Estimação 1:            Não estimado")
    print("  - Estimação 2a (sem W):  ", x_avg_2a[2])
    print("  - Estimação 2b (com W):  ", xw_avg_2b[2])
    
    print("\nBarramento 4:")
    print("  - Estimação 1:           ", x0s[2])
    print("  - Estimação 2a (sem W):  ", x_avg_2a[3])
    print("  - Estimação 2b (com W):  ", xw_avg_2b[3])
    
    print("\nBarramento 5:")
    print("  - Estimação 1:           ", x0s[3])
    print("  - Estimação 2a (sem W):  ", x_avg_2a[4])
    print("  - Estimação 2b (com W):  ", xw_avg_2b[4])
    
    # Visualização gráfica das tensões estimadas
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bus_numbers = [1, 2, 3, 4, 5]
    tensions_1 = [x0s[0], x0s[1], float('nan'), x0s[2], x0s[3]]  # Usar NaN para V3 não estimado
    tensions_2a = [x_avg_2a[0], x_avg_2a[1], x_avg_2a[2], x_avg_2a[3], x_avg_2a[4]]
    tensions_2b = [xw_avg_2b[0], xw_avg_2b[1], xw_avg_2b[2], xw_avg_2b[3], xw_avg_2b[4]]
    
    width = 0.25
    
    # Plotar barras para as magnitudes das tensões
    ax.bar([x - width for x in bus_numbers], np.abs(tensions_1), width, label='Estimação 1', alpha=0.7)
    ax.bar(bus_numbers, np.abs(tensions_2a), width, label='Estimação 2a (sem W)', alpha=0.7)
    ax.bar([x + width for x in bus_numbers], np.abs(tensions_2b), width, label='Estimação 2b (com W)', alpha=0.7)
    
    ax.set_xlabel('Número do Barramento')
    ax.set_ylabel('Magnitude da Tensão')
    ax.set_title('Comparação das Magnitudes de Tensão Estimadas')
    ax.set_xticks(bus_numbers)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adicionar texto explicativo
    plt.figtext(0.5, 0.01, 'Nota: V3 não é estimado na Estimação 1', 
                ha='center', fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    
    # Salvar figura se necessário
    if not os.path.exists('figuras'):
        os.makedirs('figuras')
    plt.savefig('figuras/comparacao_todas_tensoes.png', dpi=300)
    
    plt.show()

def plot_errors_challenge_2(ei12a, ei54a):
    """Plota os erros relativos para o desafio 2."""
    plt.figure(figsize=(10, 5))
    plt.plot(ei12a, label='Erro Relativo da Corrente I12')
    plt.plot(ei54a, label='Erro Relativo da Corrente I54')
    plt.xlabel('Iteração')
    plt.ylabel('Erro Relativo')
    plt.title('Erro Relativo das Correntes (Desafio 2 - I12=0)')
    plt.legend()
    plt.grid(True)
    
    # Salvar figura se necessário
    if not os.path.exists('figuras'):
        os.makedirs('figuras')
    plt.savefig('figuras/erro_relativo_desafio2.png', dpi=300)
    
    plt.show()


# ============================================================================================================================================
# Função para mostrar o menu e processar opções
# ============================================================================================================================================
def show_menu():
    """Exibe o menu principal e retorna a opção selecionada pelo usuário."""
    os.system('cls' if os.name == 'nt' else 'clear')  # Limpa a tela
    print("=" * 80)
    print("                 LABORATÓRIO 3 - ESTIMAÇÃO DE ESTADO                 ")
    print("=" * 80)
    print("\nEscolha uma opção:")
    print("  1. Executar 1ª Estimação de Estado (informação completa, sem V3)")
    print("  2. Executar 2ª Estimação de Estado (sem pesos, inclui V3)")
    print("  3. Executar 2ª Estimação de Estado (com pesos, inclui V3)")
    print("  4. Comparar os resultados das estimações")
    print("  5. Executar todas as estimações")
    print("  6. Avaliar combinações de pseudo-medições (Desafio 1)")
    print("  7. Comparar abordagem alternativa com abordagem completa")
    print("  8. Executar Desafio 2 - I12=0")
    print("  9. Comparar abordagem do Desafio 2 com abordagem completa")
    print("  0. Sair do programa")
    
    try:
        option = int(input("\nOpção: "))
        return option
    except ValueError:
        return -1

def main():
    # Carregar os dados
    SlackBus, Net_Info, Power_Info = load_data()
    
    # Preparar a rede
    Y, Yl, G, B, I, nBus, nLines = prepare_network(Net_Info, Power_Info, SlackBus)
    
    # Variáveis para armazenar resultados das estimações
    results = {
        'est1': None,
        'est2a': None,
        'est2b': None,
        'alt_results': None,
        'best_combo': None
    }
    
    while True:
        option = show_menu()
        
        if option == 0:
            print("\nEncerrando o programa...\n")
            sys.exit(0)
            
        elif option == 1:
            x0s, v0s = state_estimation_1(Y, Yl, I)
            results['est1'] = (x0s, v0s)
            input("\nPressione Enter para continuar...")
            
        elif option == 2:
            x0, x_avg, ei12a_2a, ei54a_2a, v, A, b0, e = state_estimation_2a(Y, Yl, I)
            plot_errors_2a(ei12a_2a, ei54a_2a)
            results['est2a'] = (x0, x_avg, ei12a_2a, ei54a_2a, v, A, b0, e)
            input("\nPressione Enter para continuar...")
            
        elif option == 3:
            if results['est2a'] is None:
                print("\nÉ necessário executar primeiro a 2ª Estimação (sem pesos)!")
                input("\nPressione Enter para continuar...")
                continue
                
            _, _, _, _, v, A, b0, e = results['est2a']
            xw0, xw_avg, ei12a_2b, ei54a_2b = state_estimation_2b(Y, A, b0, I, e, v)
            plot_errors_2b(ei12a_2b, ei54a_2b)
            results['est2b'] = (xw0, xw_avg, ei12a_2b, ei54a_2b)
            input("\nPressione Enter para continuar...")
            
        elif option == 4:
            if results['est2a'] is None or results['est2b'] is None:
                print("\nÉ necessário executar primeiro as duas versões da 2ª Estimação!")
                input("\nPressione Enter para continuar...")
                continue
                
            _, _, ei12a_2a, ei54a_2a, _, _, _, _ = results['est2a']
            _, _, ei12a_2b, ei54a_2b = results['est2b']
            compare_estimations(ei12a_2a, ei54a_2a, ei12a_2b, ei54a_2b)
            
            # Adicionar o novo gráfico de dispersão
            plot_error_scatter(ei12a_2a, ei54a_2a, ei12a_2b, ei54a_2b)
            
            # Adicionar opção para comparar todas as estimações
            print("\nDeseja também comparar os valores de tensão de todas as estimações? (s/n)")
            opt = input()
            if opt.lower() == 's':
                if results['est1'] is None:
                    print("\nÉ necessário executar também a 1ª Estimação!")
                    input("\nPressione Enter para continuar...")
                    continue
                compare_all_estimations(results)
                
            input("\nPressione Enter para continuar...")
            
        elif option == 5:
            print("\n=== Executando todas as estimações ===\n")
            
            # 1ª Estimação
            x0s, v0s = state_estimation_1(Y, Yl, I)
            results['est1'] = (x0s, v0s)
            
            # 2ª Estimação (sem pesos)
            x0, x_avg, ei12a_2a, ei54a_2a, v, A, b0, e = state_estimation_2a(Y, Yl, I)
            results['est2a'] = (x0, x_avg, ei12a_2a, ei54a_2a, v, A, b0, e)
            
            # 2ª Estimação (com pesos)
            xw0, xw_avg, ei12a_2b, ei54a_2b = state_estimation_2b(Y, A, b0, I, e, v)
            results['est2b'] = (xw0, xw_avg, ei12a_2b, ei54a_2b)
            
            # Comparar resultados
            compare_estimations(ei12a_2a, ei54a_2a, ei12a_2b, ei54a_2b)
            compare_all_estimations(results)

            # Adicionar o gráfico de dispersão
            plot_error_scatter(ei12a_2a, ei54a_2a, ei12a_2b, ei54a_2b)
            
            input("\nPressione Enter para continuar...")
        
        elif option == 6:
            print("\n=== Executando avaliação de combinações de pseudo-medições (Desafio 1) ===\n")
            
            # Avaliar todas as combinações possíveis
            alt_results, best_combo = evaluate_all_combinations(Y, Yl, I)
            results['alt_results'] = alt_results
            results['best_combo'] = best_combo
            
            
            input("\nPressione Enter para continuar...")
            
        elif option == 7:
            if results['best_combo'] is None:
                print("\nÉ necessário executar primeiro a avaliação de combinações!")
                input("\nPressione Enter para continuar...")
                continue
                
            if results['est2b'] is None:
                print("\nÉ necessário executar primeiro a 2ª Estimação (com pesos)!")
                input("\nPressione Enter para continuar...")
                continue
                
            # Comparar a melhor combinação com a abordagem completa
            compare_with_full_approach(Y, Yl, I, results['best_combo'], results['est2b'])
            
            input("\nPressione Enter para continuar...")
        
        elif option == 8:
            print("\n=== Executando as estimações para I12=0 (Desafio 2) ===\n")
            
            # Executar a análise e comparação dos cenários
            results_challenge_2 = analyze_scenario_I12_zero(Y, I, cosPhi)
            
            # Perguntar se deseja visualizar os resíduos
            print("\nDeseja visualizar a comparação dos resíduos? (s/n)")
            opt = input()
            if opt.lower() == 's':
                plot_residue_comparison(
                    results_challenge_2['branch_out']['residue'],
                    results_challenge_2['meter_fault']['residue']
                )
            
            input("\nPressione Enter para continuar...")
        
        elif option == 9:
            if results['est2b'] is None:
                print("\nÉ necessário executar primeiro a 2ª Estimação (com pesos)!")
                input("\nPressione Enter para continuar...")
                continue
                
            compare_with_full_approach_challenge_2(Y, Yl, I, results['est2b'])
            input("\nPressione Enter para continuar...")

if "__main__":
    main()

