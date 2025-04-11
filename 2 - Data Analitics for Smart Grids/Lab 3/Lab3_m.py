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
    A[2, 1] = 1                                       #   1 
    A[3, 4] = 1                                       #   1
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
def analyze_scenario_I12_zero(Y, Yl, I):
    """
    Analisa os dois possíveis cenários quando I12=0:
    1. O ramo 1-2 está fora de serviço (y12=0)
    2. O medidor de corrente está com defeito
    
    Args:
        Y: Matriz de admitância completa
        Yl: Matriz de admitância sem barramento de referência
        I: Vetor de corrente
    
    Returns:
        Dictionary com os resultados da análise
    """
    print("\n=== Análise para I12=0 (Desafio 2) ===\n")
    
    # Criar matrizes para estimação
    v = np.zeros((5), dtype=np.complex128)
    
    # Cálculo da Tensão (Referência)
    v[0:4] = 1 + np.dot(np.linalg.inv(Yl), I)
    v[4] = 1
    
    # Configuração base para as medições z
    b0 = np.zeros((8), dtype=np.complex128)
    b0[0] = 0  # *** Importante: I12=0 ***
    b0[1] = np.absolute(np.dot(Y[3, 4], (v[4] - v[3]))) * np.exp(complex(0, -1) * np.arccos(cosPhi))
    b0[2] = v[1]
    b0[3] = v[4]
    b0[4:8] = I
    
    # Começamos a analisar os dois cenários
    # 1. Ramo 1-2 fora de serviço (y12=0)
    Y_branch_out = Y.copy()
    Y_branch_out[0, 1] = 0  # Remoção do ramo 1-2
    Y_branch_out[1, 0] = 0
    
    # Ajustar matriz de admitância sem o ramo
    Y_branch_out[0, 0] -= Y[0, 1]  # Ajustar os elementos da diagonal principal
    Y_branch_out[1, 1] -= Y[0, 1]
    
    # 2. Medidor com defeito (I12=0 mas y12 permanece)
    Y_meter_fault = Y.copy()
    
    # Executar estimação de estado para os dois cenários
    results_branch_out = estimate_state_branch_out(Y_branch_out, Yl, I, v)
    results_meter_fault = estimate_state_meter_fault(Y_meter_fault, I, v, b0.copy())
    
    # Comparar os resíduos das duas estimações
    norm_branch_out = np.linalg.norm(results_branch_out['residue'])
    norm_meter_fault = np.linalg.norm(results_meter_fault['residue'])
    
    print("\n=== Resultados da Análise ===")
    print(f"Norma do resíduo para ramo 1-2 fora de serviço: {norm_branch_out:.6f}")
    print(f"Norma do resíduo para medidor com defeito: {norm_meter_fault:.6f}")
    
    # Determinar a situação mais provável
    if norm_branch_out < norm_meter_fault:
        print("\nBaseado nos resíduos, é mais provável que o ramo 1-2 esteja fora de serviço.")
        most_probable = "branch_out"
    else:
        print("\nBaseado nos resíduos, é mais provável que o medidor esteja com defeito.")
        most_probable = "meter_fault"
    
    # Retornar resultados para uso futuro
    return {
        'branch_out': results_branch_out,
        'meter_fault': results_meter_fault,
        'most_probable': most_probable
    }

def estimate_state_branch_out(Y_modified, Yl, I, v):
    """
    Estimação de estado considerando o ramo 1-2 fora de serviço (y12=0)
    
    Args:
        Y_modified: Matriz de admitância com y12=0
        Yl: Matriz de admitância sem barramento de referência
        I: Vetor de corrente
        v: Tensões de referência
    
    Returns:
        Dictionary com resultados da estimação
    """
    print("\n--- Estimação considerando ramo 1-2 fora de serviço ---")
    
    # Não precisamos da medida I12, então construímos diretamente uma matriz sem essa medida
    b0 = np.zeros((7), dtype=np.complex128)  # Apenas 7 medidas (sem I12)
    b0[0] = np.absolute(np.dot(Y_modified[3, 4], (v[4] - v[3]))) * np.exp(complex(0, -1) * np.arccos(cosPhi))
    b0[1] = v[1]
    b0[2] = v[4]
    b0[3:7] = I
    
    # Matriz Jacobiana sem a linha correspondente a I12
    A = np.zeros((7, 5), dtype=np.complex128)
    A[0, 3] = np.dot(-Y_modified[3, 4], 1)     # -Y45
    A[0, 4] = np.dot(Y_modified[3, 4], 1)      # Y45
    A[1, 1] = 1                                # 1
    A[2, 4] = 1                                # 1
    
    # Estas linhas são para as relações de potência das pseudo-medições:
    A[3, 0] = np.dot(Y_modified[0, 2], 1)      # Y13 (não temos mais Y12)
    A[3, 2] = np.dot(-Y_modified[0, 2], 1)     # -Y13
    A[4, 1] = np.dot(Y_modified[1, 2], 1)      # Y23
    A[4, 2] = np.dot(-Y_modified[1, 2], 1)     # -Y23
    A[5, 0] = A[3, 2]                          # -Y13
    A[5, 1] = A[4, 2]                          # -Y23
    A[5, 2] = np.dot(Y_modified[0, 2] + Y_modified[1, 2] + Y_modified[2, 3], 1)  # Y13+Y23+Y34
    A[5, 3] = np.dot(-Y_modified[2, 3], 1)     # -Y34
    A[6, 2] = A[5, 3]                          # -Y34
    A[6, 3] = np.dot(Y_modified[2, 3] + Y_modified[3, 4], 1)  # Y34+Y45
    A[6, 4] = np.dot(-Y_modified[3, 4], 1)     # -Y45
    
    # Matriz de pesos
    W = np.zeros((7, 7))
    np.fill_diagonal(W[0:3], 0.001**-2)  # Alto peso para medições reais
    np.fill_diagonal(W[3:7, 3:7], sig**-2)  # Peso moderado para pseudo-medições
    
    # Estimação de estado
    x = np.dot(np.dot(np.linalg.inv(np.dot(np.dot(A.conj().T, W), A)), np.dot(A.conj().T, W)), b0)
    
    # Calcular o resíduo (diferença entre medidas reais e estimadas)
    z_est = np.dot(A, x)
    residue = b0 - z_est
    
    # Mostrar resultados
    print("\nResultados com ramo 1-2 fora de serviço:")
    print("Tensão estimada no barramento 1:", x[0])
    print("Tensão estimada no barramento 2:", x[1])
    print("Tensão estimada no barramento 3:", x[2])
    print("Tensão estimada no barramento 4:", x[3])
    print("Tensão estimada no barramento 5:", x[4])
    print("\nNorma do resíduo:", np.linalg.norm(residue))
    
    return {
        'x': x,
        'residue': residue,
        'A': A,
        'W': W
    }

def estimate_state_meter_fault(Y, I, v, b0):
    """
    Estimação de estado considerando o medidor de corrente com defeito (I12=0)
    
    Args:
        Y: Matriz de admitância original
        I: Vetor de corrente
        v: Tensões de referência
        b0: Vetor de medições base com I12=0
    
    Returns:
        Dictionary com resultados da estimação
    """
    print("\n--- Estimação considerando medidor com defeito ---")
    
    # Matriz Jacobiana completa (como na estimação 2b)
    A = np.zeros((8, 5), dtype=np.complex128)
    A[0, 0] = np.dot(Y[0, 1], 1)                      #  Y12
    A[0, 1] = np.dot(-Y[0, 1], 1)                     # -Y12
    A[1, 3] = np.dot(-Y[3, 4], 1)                     # -Y45
    A[1, 4] = np.dot(Y[3, 4], 1)                      #  Y45
    A[2, 1] = 1                                       #  1 
    A[3, 4] = 1                                       #  1
    # Estas próximas linhas são para as pseudo-medições:
    A[4, 0] = np.dot(Y[0, 1] + Y[0, 2], 1)            #  Y12+Y13
    A[4, 1] = A[0, 1]                                 # -Y12
    A[4, 2] = np.dot(-Y[0, 2], 1)                     # -Y13
    A[5, 0] = A[0, 1]                                 # -Y12
    A[5, 1] = np.dot(Y[0, 1] + Y[1,2], 1)             #  Y12+Y23
    A[5, 2] = np.dot(-Y[1, 2], 1)                     # -Y23
    A[6, 0] = A[4, 2]                                 # -Y13
    A[6, 1] = A[5, 2]                                 # -Y23
    A[6, 2] = np.dot(Y[0, 2] + Y[1, 2] + Y[2, 3], 1)  #  Y13+Y23+Y34
    A[6, 3] = np.dot(-Y[2, 3], 1)                     # -Y34
    A[7, 2] = A[6, 3]                                 # -Y34
    A[7, 3] = np.dot(Y[2, 3] + Y[3, 4], 1)            #  Y34+Y45
    A[7, 4] = A[1, 3]                                 # -Y45
    
    # Matriz de pesos
    W = np.zeros((8, 8))
    np.fill_diagonal(W[0:4], 0.001**-2)  # Alto peso para medidas reais
    np.fill_diagonal(W[4:8, 4:8], sig**-2)  # Peso moderado para pseudo-medidas
    
    # Estimação de estado
    x = np.dot(np.dot(np.linalg.inv(np.dot(np.dot(A.conj().T, W), A)), np.dot(A.conj().T, W)), b0)
    
    # Calcular o resíduo
    z_est = np.dot(A, x)
    residue = b0 - z_est
    
    # Mostrar resultados
    print("\nResultados com medidor com defeito:")
    print("Tensão estimada no barramento 1:", x[0])
    print("Tensão estimada no barramento 2:", x[1])
    print("Tensão estimada no barramento 3:", x[2])
    print("Tensão estimada no barramento 4:", x[3])
    print("Tensão estimada no barramento 5:", x[4])
    print("\nNorma do resíduo:", np.linalg.norm(residue))
    
    return {
        'x': x,
        'residue': residue,
        'A': A,
        'W': W
    }


    """
    Compara os resultados do Desafio 2 com a abordagem completa (estimação 2b regular).
    
    Args:
        Y: Matriz de admitância completa
        Yl: Matriz de admitância sem barramento de referência  
        I: Vetor de corrente
        results_2b: Resultados da estimação 2b regular
    """
    print("\n=== Comparação das Estimações do Desafio 2 com a Abordagem Regular ===\n")
    
    # Executa a análise do desafio 2
    results_challenge2 = analyze_scenario_I12_zero(Y, Yl, I)
    
    # Extrai a estimação mais provável
    most_probable = results_challenge2['most_probable']
    x_challenge2 = results_challenge2[most_probable]['x']
    
    # Extrai os resultados da estimação 2b regular
    _, x_regular, _, _ = results_2b
    
    # Comparar as magnitudes das tensões estimadas
    bus_numbers = [1, 2, 3, 4, 5]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    width = 0.35
    
    ax.bar([x - width/2 for x in bus_numbers], np.abs(x_challenge2), width, 
           label=f'Estimação para I12=0 ({most_probable.replace("_", " ").title()})', alpha=0.7)
    ax.bar([x + width/2 for x in bus_numbers], np.abs(x_regular), width, 
           label='Estimação Regular (2b)', alpha=0.7)
    
    # Calcular diferenças percentuais
    diff_pct = []
    for i in range(5):
        mag_challenge2 = np.abs(x_challenge2[i])
        mag_regular = np.abs(x_regular[i])
        diff = abs(mag_challenge2 - mag_regular) / mag_regular * 100
        diff_pct.append(diff)
    
    # Adicionar texto com as diferenças
    for i, bus in enumerate(bus_numbers):
        plt.annotate(f"{diff_pct[i]:.2f}%", 
                    xy=(bus, max(np.abs(x_challenge2[i]), np.abs(x_regular[i])) + 0.02),
                    ha='center')
    
    ax.set_xlabel('Número do Barramento')
    ax.set_ylabel('Magnitude da Tensão')
    ax.set_title('Comparação das Magnitudes de Tensão entre Desafio 2 e Estimação Regular')
    ax.set_xticks(bus_numbers)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Salvar figura
    if not os.path.exists('figuras'):
        os.makedirs('figuras')
    plt.savefig('figuras/comparacao_desafio2_regular.png', dpi=300)
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar diferenças percentuais na saída de texto
    print("\nDiferença percentual nas magnitudes de tensão:")
    for i, bus in enumerate(bus_numbers):
        print(f"Barramento {bus}: {diff_pct[i]:.2f}%")

def compare_weight_matrices_effect(Y, Yl, I):
    """
    Compara o efeito de diferentes matrizes de pesos na estimação de estado 
    para os dois cenários quando I12=0.
    
    Args:
        Y: Matriz de admitância completa
        Yl: Matriz de admitância sem barramento de referência
        I: Vetor de corrente
    
    Returns:
        Dictionary com os resultados da análise
    """
    print("\n=== Análise do Efeito das Matrizes de Peso (I12=0) ===\n")
    
    # Criar matrizes para estimação
    v = np.zeros((5), dtype=np.complex128)
    
    # Cálculo da Tensão (Referência)
    v[0:4] = 1 + np.dot(np.linalg.inv(Yl), I)
    v[4] = 1
    
    # Configuração base para as medições z
    b0 = np.zeros((8), dtype=np.complex128)
    b0[0] = 0  # *** Importante: I12=0 ***
    b0[1] = np.absolute(np.dot(Y[3, 4], (v[4] - v[3]))) * np.exp(complex(0, -1) * np.arccos(cosPhi))
    b0[2] = v[1]
    b0[3] = v[4]
    b0[4:8] = I
    
    # Cenário 1: Ramo 1-2 fora de serviço (y12=0)
    Y_branch_out = Y.copy()
    Y_branch_out[0, 1] = 0
    Y_branch_out[1, 0] = 0
    Y_branch_out[0, 0] -= Y[0, 1]
    Y_branch_out[1, 1] -= Y[0, 1]
    
    # Cenário 2: Medidor com defeito
    Y_meter_fault = Y.copy()
    
    # Definir três matrizes de pesos diferentes
    weight_configs = {
        "Matriz Identidade": {
            "description": "Pesos iguais para medições reais e pseudo-medições",
            "branch_out": np.eye(7),  # Identidade para branch_out (matriz 7x7)
            "meter_fault": np.eye(8)  # Identidade para meter_fault (matriz 8x8)
        },
        "Maior Peso para Medições Reais": {
            "description": "Pesos maiores para medições reais (10x) que para pseudo-medições",
            "branch_out": np.zeros((7, 7)),
            "meter_fault": np.zeros((8, 8))
        },
        "Peso Muito Maior para Medições Reais": {
            "description": "Pesos muito maiores para medições reais (1000x) que para pseudo-medições",
            "branch_out": np.zeros((7, 7)),
            "meter_fault": np.zeros((8, 8))
        }
    }
    
    # Configurar as matrizes de peso específicas (não-identidade)
    # Matriz 2: Maior peso para medições reais
    np.fill_diagonal(weight_configs["Maior Peso para Medições Reais"]["branch_out"][0:3], 10)  
    np.fill_diagonal(weight_configs["Maior Peso para Medições Reais"]["branch_out"][3:7, 3:7], 1)
    
    np.fill_diagonal(weight_configs["Maior Peso para Medições Reais"]["meter_fault"][0:4], 10)
    np.fill_diagonal(weight_configs["Maior Peso para Medições Reais"]["meter_fault"][4:8, 4:8], 1)
    
    # Matriz 3: Peso muito maior para medições reais
    np.fill_diagonal(weight_configs["Peso Muito Maior para Medições Reais"]["branch_out"][0:3], 1000)
    np.fill_diagonal(weight_configs["Peso Muito Maior para Medições Reais"]["branch_out"][3:7, 3:7], 1)
    
    np.fill_diagonal(weight_configs["Peso Muito Maior para Medições Reais"]["meter_fault"][0:4], 1000)
    np.fill_diagonal(weight_configs["Peso Muito Maior para Medições Reais"]["meter_fault"][4:8, 4:8], 1)
    
    # Armazenar os resultados para cada configuração
    results = {}
    norm_branch_out_values = []
    norm_meter_fault_values = []
    weight_labels = []
    
    # Para cada configuração de pesos
    for weight_name, config in weight_configs.items():
        print(f"\n== Testando {weight_name} ==")
        print(f"Descrição: {config['description']}")
        
        # Matrizes de peso para cada cenário
        W_branch_out = config["branch_out"]
        W_meter_fault = config["meter_fault"]
        
        # Executar estimação para o cenário branch_out
        results_branch = estimate_state_with_weights(Y_branch_out, I, v, "branch_out", W_branch_out)
        
        # Executar estimação para o cenário meter_fault
        results_meter = estimate_state_with_weights(Y_meter_fault, I, v, "meter_fault", W_meter_fault, b0)
        
        # Calcular e armazenar normas de resíduo
        norm_branch = np.linalg.norm(results_branch['residue'])
        norm_meter = np.linalg.norm(results_meter['residue'])
        
        print(f"Norma do resíduo para ramo 1-2 fora de serviço: {norm_branch:.6f}")
        print(f"Norma do resíduo para medidor com defeito: {norm_meter:.6f}")
        
        # Determinar o cenário mais provável
        if norm_branch < norm_meter:
            most_probable = "branch_out"
            print(f"Com {weight_name}, é mais provável que o ramo 1-2 esteja fora de serviço.")
        else:
            most_probable = "meter_fault"
            print(f"Com {weight_name}, é mais provável que o medidor esteja com defeito.")
        
        # Armazenar resultados
        results[weight_name] = {
            "branch_out": results_branch,
            "meter_fault": results_meter,
            "norm_branch_out": norm_branch,
            "norm_meter_fault": norm_meter,
            "most_probable": most_probable
        }
        
        # Adicionar valores para o gráfico
        norm_branch_out_values.append(norm_branch)
        norm_meter_fault_values.append(norm_meter)
        weight_labels.append(weight_name.replace(" ", "\n"))
    
    # Plotar comparação dos resíduos
    plot_weight_matrix_comparison(weight_labels, norm_branch_out_values, norm_meter_fault_values)
    
    return results

def estimate_state_with_weights(Y_modified, I, v, scenario_type, W, b0=None):
    """
    Executa a estimação de estado com uma matriz de pesos específica.
    
    Args:
        Y_modified: Matriz de admitância (modificada de acordo com o cenário)
        I: Vetor de corrente
        v: Tensões de referência
        scenario_type: Tipo de cenário ('branch_out' ou 'meter_fault')
        W: Matriz de pesos a ser utilizada
        b0: Vetor de medições (apenas necessário para meter_fault)
    
    Returns:
        Dictionary com resultados da estimação
    """
    if scenario_type == "branch_out":
        # Não precisamos da medida I12, similar ao estimate_state_branch_out
        b0_local = np.zeros((7), dtype=np.complex128)
        b0_local[0] = np.absolute(np.dot(Y_modified[3, 4], (v[4] - v[3]))) * np.exp(complex(0, -1) * np.arccos(cosPhi))
        b0_local[1] = v[1]
        b0_local[2] = v[4]
        b0_local[3:7] = I
        
        # Matriz Jacobiana sem a linha correspondente a I12
        A = np.zeros((7, 5), dtype=np.complex128)
        A[0, 3] = np.dot(-Y_modified[3, 4], 1)
        A[0, 4] = np.dot(Y_modified[3, 4], 1)
        A[1, 1] = 1
        A[2, 4] = 1
        
        # Relações de potência
        A[3, 0] = np.dot(Y_modified[0, 2], 1)
        A[3, 2] = np.dot(-Y_modified[0, 2], 1)
        A[4, 1] = np.dot(Y_modified[1, 2], 1)
        A[4, 2] = np.dot(-Y_modified[1, 2], 1)
        A[5, 0] = A[3, 2]
        A[5, 1] = A[4, 2]
        A[5, 2] = np.dot(Y_modified[0, 2] + Y_modified[1, 2] + Y_modified[2, 3], 1)
        A[5, 3] = np.dot(-Y_modified[2, 3], 1)
        A[6, 2] = A[5, 3]
        A[6, 3] = np.dot(Y_modified[2, 3] + Y_modified[3, 4], 1)
        A[6, 4] = np.dot(-Y_modified[3, 4], 1)
    
    else:  # meter_fault
        # Usar o vetor de medições fornecido
        b0_local = b0.copy()
        
        # Matriz Jacobiana completa
        A = np.zeros((8, 5), dtype=np.complex128)
        A[0, 0] = np.dot(Y_modified[0, 1], 1)
        A[0, 1] = np.dot(-Y_modified[0, 1], 1)
        A[1, 3] = np.dot(-Y_modified[3, 4], 1)
        A[1, 4] = np.dot(Y_modified[3, 4], 1)
        A[2, 1] = 1
        A[3, 4] = 1
        
        # Relações de potência
        A[4, 0] = np.dot(Y_modified[0, 1] + Y_modified[0, 2], 1)
        A[4, 1] = A[0, 1]
        A[4, 2] = np.dot(-Y_modified[0, 2], 1)
        A[5, 0] = A[0, 1]
        A[5, 1] = np.dot(Y_modified[0, 1] + Y_modified[1,2], 1)
        A[5, 2] = np.dot(-Y_modified[1, 2], 1)
        A[6, 0] = A[4, 2]
        A[6, 1] = A[5, 2]
        A[6, 2] = np.dot(Y_modified[0, 2] + Y_modified[1, 2] + Y_modified[2, 3], 1)
        A[6, 3] = np.dot(-Y_modified[2, 3], 1)
        A[7, 2] = A[6, 3]
        A[7, 3] = np.dot(Y_modified[2, 3] + Y_modified[3, 4], 1)
        A[7, 4] = A[1, 3]
    
    # Estimação de estado
    x = np.dot(np.dot(np.linalg.inv(np.dot(np.dot(A.conj().T, W), A)), np.dot(A.conj().T, W)), b0_local)
    
    # Calcular o resíduo
    z_est = np.dot(A, x)
    residue = b0_local - z_est
    
    return {
        'x': x,
        'residue': residue,
        'A': A
    }

def plot_weight_matrix_comparison(weight_labels, norm_branch_out_values, norm_meter_fault_values):
    """
    Plota a comparação das normas de resíduo para diferentes matrizes de peso.
    
    Args:
        weight_labels: Rótulos para as matrizes de peso
        norm_branch_out_values: Valores da norma para o cenário branch_out
        norm_meter_fault_values: Valores da norma para o cenário meter_fault
    """
    plt.figure(figsize=(10, 6))
    x = np.arange(len(weight_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bar1 = ax.bar(x - width/2, norm_branch_out_values, width, label='Ramo 1-2 Fora de Serviço', alpha=0.7)
    bar2 = ax.bar(x + width/2, norm_meter_fault_values, width, label='Medidor com Defeito', alpha=0.7)
    
    # Destaque para o menor resíduo em cada configuração
    for i, (branch, meter) in enumerate(zip(norm_branch_out_values, norm_meter_fault_values)):
        if branch < meter:
            bar1[i].set_color('green')
            bar1[i].set_alpha(1.0)
        else:
            bar2[i].set_color('green')
            bar2[i].set_alpha(1.0)
    
    ax.set_xlabel('Matriz de Pesos')
    ax.set_ylabel('Norma do Resíduo')
    ax.set_title('Comparação das Normas de Resíduo para Diferentes Matrizes de Peso')
    ax.set_xticks(x)
    ax.set_xticklabels(weight_labels)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adicionar valores nas barras
    for i, v in enumerate(zip(norm_branch_out_values, norm_meter_fault_values)):
        ax.text(i - width/2, v[0] + 0.01, f"{v[0]:.4f}", ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, v[1] + 0.01, f"{v[1]:.4f}", ha='center', va='bottom', fontsize=9)
    
    # Salvar a figura
    if not os.path.exists('figuras'):
        os.makedirs('figuras')
    plt.savefig('figuras/comparacao_matrizes_peso_i12_zero.png', dpi=300)
    
    plt.tight_layout()
    plt.show()

def plot_residue_comparison(residue_branch_out, residue_meter_fault):
    """
    Plota a comparação dos resíduos individuais para os dois cenários.
    
    Args:
        residue_branch_out: Vetor de resíduo para o cenário branch_out
        residue_meter_fault: Vetor de resíduo para o cenário meter_fault
    """
    # Ajustar os tamanhos para comparação
    len_branch = len(residue_branch_out)
    len_meter = len(residue_meter_fault)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plotar os valores absolutos dos resíduos
    branch_abs = np.abs(residue_branch_out)
    meter_abs = np.abs(residue_meter_fault)
    
    # Primeiro subplot: Branch Out
    x_branch = np.arange(len_branch)
    ax1.bar(x_branch, branch_abs, alpha=0.7, color='blue')
    ax1.set_title('Resíduos Absolutos - Ramo 1-2 Fora de Serviço')
    ax1.set_ylabel('Valor Absoluto')
    ax1.set_xticks(x_branch)
    ax1.set_xticklabels([f"z{i+1}" for i in range(len_branch)])
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Segundo subplot: Meter Fault
    x_meter = np.arange(len_meter)
    ax2.bar(x_meter, meter_abs, alpha=0.7, color='red')
    ax2.set_title('Resíduos Absolutos - Medidor com Defeito')
    ax2.set_ylabel('Valor Absoluto')
    ax2.set_xlabel('Elemento do Vetor de Medições')
    ax2.set_xticks(x_meter)
    ax2.set_xticklabels([f"z{i+1}" for i in range(len_meter)])
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Adicionar informação da norma
    norm_branch = np.linalg.norm(residue_branch_out)
    norm_meter = np.linalg.norm(residue_meter_fault)
    
    ax1.text(0.02, 0.95, f"Norma do resíduo: {norm_branch:.6f}", 
             transform=ax1.transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    
    ax2.text(0.02, 0.95, f"Norma do resíduo: {norm_meter:.6f}",
             transform=ax2.transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Destacar qual tem menor norma
    if norm_branch < norm_meter:
        fig.suptitle("Análise de Resíduos (Ramo 1-2 Fora de Serviço é mais provável)", fontsize=16)
    else:
        fig.suptitle("Análise de Resíduos (Medidor com Defeito é mais provável)", fontsize=16)
    
    # Salvar a figura
    if not os.path.exists('figuras'):
        os.makedirs('figuras')
    plt.savefig('figuras/comparacao_residuos_detalhados.png', dpi=300)
    
    plt.tight_layout()
    plt.show()


# ============================================================================================================================================
# Funções para o Desafio 3 - estimação de estado para vários valores de sigma
# ============================================================================================================================================
def state_estimation_sigma_2a(Y, Yl, I, sigmas):
    """
    2ª Estimação de Estado para vários valores de sigma.
    
    Args:
        Y: Matriz de admitância completa
        Yl: Matriz de admitância sem barramento de referência
        I: Vetor de corrente
        sigmas: Lista de valores de sigma para os quais a estimação será realizada
    
    Returns:
        Resultados da estimação para cada valor de sigma
    """
    print("\n=== 2ª Estimação de Estado para Vários Valores de Sigma (a) ===\n")
    
    # Criação da Matriz
    b0 = np.zeros((8), dtype=np.complex128)
    A = np.zeros((8, 5), dtype=np.complex128)
    v = np.zeros((5), dtype=np.complex128)
    
    # Cálculo da Tensão (Referência)
    v[0:4] = 1 + np.dot(np.linalg.inv(Yl), I)
    v[4] = 1
    
    # Valores de medição z 
    b0[0] = np.dot(np.absolute(np.dot(Y[0, 1], (v[0] - v[1])), np.exp(complex(0, -1) * np.arccos(cosPhi))))
    b0[1] = np.dot(np.absolute(np.dot(Y[3, 4], (1 - v[3])), np.exp(complex(0, -1) * np.arccos(cosPhi))))
    b0[2] = v[1]
    b0[3] = 1
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

    # Ruído a ser adicionado às pseudo-medições
    np.random.seed(42)
    
    results = {}
    
    for sig in sigmas:
        print(f"Processando sigma = {sig}")
        
        # Gerar ruído específico para este sigma
        e = np.random.normal(0.0, 1.0, size=(4, m)) * sig
        
        # Estimação sem consideração do ruído 
        x0 = np.dot(np.dot(np.linalg.inv(np.dot(A.conj().T, A)), A.conj().T), b0)
        
        # Criação da Matriz
        sx = np.zeros(5, dtype=np.complex128)
        rms = np.zeros((5, m))
        ei12a = np.zeros(m)
        ei54a = np.zeros(m)
        
        # Para cada iteração
        for i in range(m):
            # Copiar o vetor de medição
            b1 = np.copy(b0)
            
            # Adicionar ruído apenas às pseudo-medições
            b1[4:8] = I + e[:, i]
            
            # Estimar tensões
            x = np.dot(np.dot(np.linalg.inv(np.dot(A.conj().T, A)), A.conj().T), b1)
            
            # Acumular para médias
            sx = sx + x
            
            # Erros nas tensões
            rms[:, i] = np.sqrt(np.abs(np.dot((x - x0), np.conjugate(x - x0))))
            
            # Erros relativos da corrente
            ei12a[i] = np.divide(
                np.absolute(np.absolute(np.dot(Y[0, 1], (x[0] - x[1]))) - np.absolute(np.dot(Y[0, 1], (v[0] - v[1])))),
                np.absolute(np.dot(Y[0, 1], (v[0] - v[1])))
            )
            ei54a[i] = np.divide(
                np.absolute(np.absolute(np.dot(Y[3, 4], (1 - x[3]))) - np.absolute(np.dot(Y[3, 4], (1 - v[3])))),
                np.absolute(np.dot(Y[3, 4], (1 - v[3])))
            )
        
        # Cálculo de médias
        x_avg = sx / m
        ee = np.sum(rms, axis=1) / m
        
        # Armazenar resultados para este sigma
        results[sig] = (x0, x_avg, ei12a, ei54a, ee)
        
        print(f"Tensão média estimada (sigma={sig}):")
        print(f"V1 = {x_avg[0]}")
        print(f"V2 = {x_avg[1]}")
        print(f"V3 = {x_avg[2]}")
        print(f"V4 = {x_avg[3]}")
        print(f"V5 = {x_avg[4]}")
        print(f"Erro RMS médio: {ee}\n")
    
    return results

def state_estimation_sigma_2b(Y, Yl, I, sigmas):
    """
    2ª Estimação de Estado para vários valores de sigma.
    
    Args:
        Y: Matriz de admitância completa
        Yl: Matriz de admitância sem barramento de referência
        I: Vetor de corrente
        sigmas: Lista de valores de sigma para os quais a estimação será realizada
    
    Returns:
        Resultados da estimação para cada valor de sigma
    """
    print("\n=== 2ª Estimação de Estado para Vários Valores de Sigma (b) ===\n")
    
    # Criação da Matriz
    b0 = np.zeros((8), dtype=np.complex128)
    A = np.zeros((8, 5), dtype=np.complex128)
    v = np.zeros((5), dtype=np.complex128)
    
    # Cálculo da Tensão (Referência)
    v[0:4] = 1 + np.dot(np.linalg.inv(Yl), I)
    v[4] = 1
    
    # Valores de medição z 
    b0[0] = np.dot(np.absolute(np.dot(Y[0, 1], (v[0] - v[1])), np.exp(complex(0, -1) * np.arccos(cosPhi))))
    b0[1] = np.dot(np.absolute(np.dot(Y[3, 4], (1 - v[3])), np.exp(complex(0, -1) * np.arccos(cosPhi))))
    b0[2] = v[1]
    b0[3] = 1
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

    # Ruído a ser adicionado às pseudo-medições
    np.random.seed(42)
    e = np.random.normal(0.0, 1.0, size=(4, m)) * sigmas

    # Criação da Matriz
    sx = np.zeros(5, dtype=np.complex128)  # CORREÇÃO: Usar complex128 para evitar erros de tipo
    rms = np.zeros((5, m))
    ei12a = np.zeros(m)
    ei54a = np.zeros(m)

    results = {}

    for sig in sigmas:
        # Criação da Matriz
        b1 = np.zeros((8), dtype=np.complex128)
        b1 = np.copy(b0)

        # Matriz dos Pesos (W) 
        W = np.zeros((len(b1), len(b1)))

        # Medições reais (primeiros 4 elementos) recebem peso alto = maior confiança
        np.fill_diagonal(W[0:4], 0.001**-2)
        np.fill_diagonal(W[4:8,4:8], sig**-2)   

        # Estimação sem considerar o ruído, mas considerando o peso 
        xw0 = np.dot(np.dot(np.linalg.inv(np.dot(np.dot(A.conjugate().T, W), A)), np.dot(A.conjugate().T, W)), b1)

        # Criação da Matriz
        sx = np.zeros(5, dtype=np.complex128)
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
                np.absolute(np.dot(Y[0, 1], (v[0] - v[1]))
            ))
            ei54a[i] = np.divide(
                np.absolute(np.absolute(np.dot(Y[3, 4], (1 - xw[3]))) - np.absolute(np.dot(Y[3, 4], (1 - v[3])))),
                np.absolute(np.dot(Y[3, 4], (1 - v[3])
            )))

        # Estimação média da tensão
        xw_avg = sx / m

        # Erro RMS médio da tensão
        eew = np.sum(rms, axis=1) / m

        # Salvar resultados
        results[sig] = (xw0, xw_avg, ei12a, ei54a, eew)

    return results

def plot_state_estimation_sigma(results):
    """
    Plotar resultados da estimação de estado para vários valores de sigma.
    
    Args:
        results: Dicionário de resultados da estimação para cada valor de sigma

    Returns:
        Gráficos dos resultados
    """
    print("\n=== Plotar Resultados da Estimação de Estado para Vários Valores de Sigma ===\n")

    # Preparar gráficos
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Estimação de Estado para Vários Valores de Sigma", fontsize=16)

    # Gráfico 1: Tensão Média Estimada no Barramento 1
    axs[0, 0].set_title("Tensão Média Estimada no Barramento 1")
    axs[0, 0].set_xlabel("Sigma")
    axs[0, 0].set_ylabel("Magnitude")
    axs[0, 0].plot(results.keys(), [np.abs(r[1][0]) for r in results.values()], marker='o')

    # Gráfico 2: Tensão Média Estimada no Barramento 2
    axs[0, 1].set_title("Tensão Média Estimada no Barramento 2")
    axs[0, 1].set_xlabel("Sigma")
    axs[0, 1].set_ylabel("Magnitude")
    axs[0, 1].plot(results.keys(), [np.abs(r[1][1]) for r in results.values()], marker='o')

    # Gráfico 3: Tensão Média Estimada no Barramento 4
    axs[1, 0].set_title("Tensão Média Estimada no Barramento 4")
    axs[1, 0].set_xlabel("Sigma")
    axs[1, 0].set_ylabel("Magnitude")
    axs[1, 0].plot(results.keys(), [np.abs(r[1][3]) for r in results.values()], marker='o')

    # Gráfico 4: Tensão Média Estimada no Barramento 5
    axs[1, 1].set_title("Tensão Média Estimada no Barramento 5")
    axs[1, 1].set_xlabel("Sigma")
    axs[1, 1].set_ylabel("Magnitude")
    axs[1, 1].plot(results.keys(), [np.abs(r[1][4]) for r in results.values()], marker='o')

    plt.tight_layout()
    plt.show()

    # Gráfico 5: Erro RMS Médio da Tensão
    plt.figure(figsize=(10, 5))
    plt.title("Erro RMS Médio da Tensão")
    plt.xlabel("Sigma")
    plt.ylabel("Erro RMS")
    plt.plot(results.keys(), [r[4] for r in results.values()], marker='o')
    plt.show()

    return

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
    print("  10. Executar estimação para vários valores de sigma (sem pesos)")
    print("  11. Executar estimação para vários valores de sigma (com pesos)")
    print("  12. Plotar resultados da estimação para vários valores de sigma")
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
            
            # 1. Run basic analysis first
            print("\n--- Análise Básica dos Cenários ---")
            results_challenge_2 = analyze_scenario_I12_zero(Y, Yl, I)

            # 2. Run weighted analysis with different weight types and compare results
            print("\n--- Análise com Diferentes Matrizes de Pesos ---")
            weight_comparison = compare_weight_matrices_effect(Y, Yl, I)
            
            # 3. Ask if user wants to see residue comparison from basic analysis
            print("\nDeseja visualizar a comparação dos resíduos da análise básica? (s/n)")
            opt = input()
            if opt.lower() == 's':
                plot_residue_comparison(
                    results_challenge_2['branch_out']['residue'],
                    results_challenge_2['meter_fault']['residue']
                )
            
            # 4. Store results for potential future comparisons
            results['challenge_2'] = {
                'basic': results_challenge_2,
                'weighted': weight_comparison
            }
            
            input("\nPressione Enter para continuar...")
        
        elif option == 9:
            if results['est2b'] is None:
                print("\nÉ necessário executar primeiro a 2ª Estimação (com pesos)!")
                input("\nPressione Enter para continuar...")
                continue
                
            compare_with_full_approach_challenge_2(Y, Yl, I, results['est2b'])
            input("\nPressione Enter para continuar...")

        elif option == 10:
            # Solicitar valores de sigma ao usuário
            try:
                sigma_min = float(input("\nValor mínimo de sigma (ex: 0.1): "))
                sigma_max = float(input("Valor máximo de sigma (ex: 1.0): "))
                sigma_steps = int(input("Número de passos (ex: 10): "))
                
                sigmas = np.linspace(sigma_min, sigma_max, sigma_steps)
                print(f"\nValores de sigma a serem utilizados: {sigmas}")
                
                results['sigma_2a'] = state_estimation_sigma_2a(Y, Yl, I, sigmas)
                print("\nEstimação concluída para todos os valores de sigma!")
                
                # Perguntar se deseja plotar os resultados imediatamente
                if input("\nDeseja plotar os resultados agora? (s/n): ").lower() == 's':
                    plot_state_estimation_sigma(results['sigma_2a'])
            except ValueError:
                print("\nEntrada inválida. Certifique-se de inserir valores numéricos corretos.")
            
            input("\nPressione Enter para continuar...")
            
        elif option == 11:
            # Solicitar valores de sigma ao usuário
            try:
                sigma_min = float(input("\nValor mínimo de sigma (ex: 0.1): "))
                sigma_max = float(input("Valor máximo de sigma (ex: 1.0): "))
                sigma_steps = int(input("Número de passos (ex: 10): "))
                
                sigmas = np.linspace(sigma_min, sigma_max, sigma_steps)
                print(f"\nValores de sigma a serem utilizados: {sigmas}")
                
                results['sigma_2b'] = state_estimation_sigma_2b(Y, Yl, I, sigmas)
                print("\nEstimação concluída para todos os valores de sigma!")
                
                # Perguntar se deseja plotar os resultados imediatamente
                if input("\nDeseja plotar os resultados agora? (s/n): ").lower() == 's':
                    plot_state_estimation_sigma(results['sigma_2b'])
            except ValueError:
                print("\nEntrada inválida. Certifique-se de inserir valores numéricos corretos.")
            
            input("\nPressione Enter para continuar...")
            
        elif option == 12:
            # Verificar quais estimações com diferentes sigmas foram executadas
            if results['sigma_2a'] is not None and results['sigma_2b'] is not None:
                escolha = input("\nQual estimação deseja plotar? (a - sem pesos, b - com pesos): ").lower()
                if escolha == 'a':
                    plot_state_estimation_sigma(results['sigma_2a'])
                elif escolha == 'b':
                    plot_state_estimation_sigma(results['sigma_2b'])
                else:
                    print("\nEscolha inválida!")
            elif results['sigma_2a'] is not None:
                print("\nPlotando resultados da estimação sem pesos (a)...")
                plot_state_estimation_sigma(results['sigma_2a'])
            elif results['sigma_2b'] is not None:
                print("\nPlotando resultados da estimação com pesos (b)...")
                plot_state_estimation_sigma(results['sigma_2b'])
            else:
                print("\nÉ necessário executar a estimação para diferentes valores de sigma primeiro (opções 10 ou 11)!")
            
            input("\nPressione Enter para continuar...")

if "__main__":
    main()

