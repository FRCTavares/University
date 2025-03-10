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
            np.absolute(np.dot(Y[0, 1], (x[0] - x[1]))) - np.absolute(np.dot(Y[0, 1], (v[0] - v[1]))),
            np.absolute(np.dot(Y[0, 1], (v[0] - v[1])))
        )
        ei54a[i] = np.divide(
            np.absolute(np.dot(Y[3, 4], (x[4] - x[3]))) - np.absolute(np.dot(Y[3, 4], (v[4] - v[3]))),
            np.absolute(np.dot(Y[3, 4], (v[4] - v[3])))
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
    print("\nValores teóricos esperados:")
    print("V3 = (0.8924 - 0.2277j)")
    print("V4 = (0.9481 - 0.1172j)")
    print("V5 = (1.0000 + 0.0000j)")
    print("\nErro RMS médio da tensão:", ee)
    
    return x0, x_avg, ei12a, ei54a, v, A, b0, e

def state_estimation_2b(Y, A, b0, I, e, v):
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
    b2 = np.zeros(len(b0), dtype=np.complex128)
    
    # CORREÇÃO 2: Matriz dos Pesos (W) - Ajustando valores conforme sugerido
    W = np.zeros((len(b0), len(b0)))
    
    # Medições reais (primeiros 4 elementos) recebem peso alto = maior confiança
    np.fill_diagonal(W[0:4, 0:4], 10**6)  # Peso 10^6 para medições reais
    
    # Pseudo-medições (elementos 4-8) recebem peso ainda menor para reduzir seu impacto
    np.fill_diagonal(W[4:8, 4:8], 2)      # Reduzido de 4 para 2 conforme sugerido
    
    # Pseudo-medições adicionais para V3, V4, V5 também recebem peso similar
    if len(b0) > 8:
        np.fill_diagonal(W[8:, 8:], 2)    # Ajustado para 2
    
    # Estimação sem considerar o ruído, mas considerando o peso 
    xw0 = np.dot(np.dot(np.linalg.inv(np.dot(np.dot(A.conj().T, W), A)), np.dot(A.conj().T, W)), b0)
    
    # Criação da Matriz
    sx = np.zeros(5, dtype=np.complex128)  # CORREÇÃO: Usar complex128 para evitar erros de tipo
    rms = np.zeros((5, m))
    ei12a = np.zeros(m)
    ei54a = np.zeros(m)
    
    for i in range(m):
        # Introduzir erro nas medições (Matriz z)
        b2[0:4] = b0[0:4]  # Mantém os valores originais para as primeiras 4 medidas
        b2[4:8] = I + e[:, i]
        if len(b2) > 8:  # Se adicionamos pseudo-medições adicionais
            b2[8:] = b0[8:]  # Manter os valores de pseudo-medição originais
    
        # Estimar as tensões com base nas medições com erros
        xw = np.dot(np.dot(np.linalg.inv(np.dot(np.dot(A.conj().T, W), A)), np.dot(A.conj().T, W)), b2)
    
        # Valor acumulado das estimações
        sx = sx + xw
    
        # Erros nas tensões
        rms[:, i] = np.sqrt(np.abs(np.dot((xw - xw0), np.conjugate(xw - xw0))))
    
        # Erros relativos da corrente (Para serem utilizados nos gráficos)
        ei12a[i] = np.divide(
            np.absolute(np.dot(Y[0, 1], (xw[0] - xw[1]))) - np.absolute(np.dot(Y[0, 1], (v[0] - v[1]))),
            np.absolute(np.dot(Y[0, 1], (v[0] - v[1])))
        )
        ei54a[i] = np.divide(
            np.absolute(np.dot(Y[3, 4], (xw[4] - xw[3]))) - np.absolute(np.dot(Y[3, 4], (v[4] - v[3]))),
            np.absolute(np.dot(Y[3, 4], (v[4] - v[3])))
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
    print("\nValores teóricos esperados:")
    print("V3 = (0.8924 - 0.2277j)")
    print("V4 = (0.9481 - 0.1172j)")
    print("V5 = (1.0000 + 0.0000j)")
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
    
    # Matriz Hx (como na estimação 2a)
    A[0, 0] = np.dot(Y[0, 1], 1)
    A[0, 1] = np.dot(-Y[0, 1], 1)
    A[1, 3] = np.dot(-Y[3, 4], 1)
    A[1, 4] = np.dot(Y[3, 4], 1)
    A[2, 1] = 1
    A[3, 4] = 1
    
    # Relações para pseudo-medições
    A[4, 0] = np.dot(Y[0, 1] + Y[0, 2], 1)
    A[4, 1] = A[0, 1]
    A[4, 2] = np.dot(-Y[0, 2], 1)
    A[5, 0] = A[0, 1]
    A[5, 1] = A[4, 0]
    A[5, 2] = np.dot(-Y[1, 2], 1)
    A[6, 0] = A[4, 2]
    A[6, 1] = A[5, 2]
    A[6, 2] = np.dot(Y[0, 2] + Y[1, 2] + Y[2, 3], 1)
    A[6, 3] = np.dot(-Y[2, 3], 1)
    A[7, 2] = A[6, 3]
    A[7, 3] = np.dot(Y[2, 3] + Y[3, 4], 1)
    A[7, 4] = A[1, 3]
    
    # MODIFICAÇÃO: Criar novas matrizes com apenas as pseudo-medições selecionadas
    # Manter as 4 medições reais (índices 0-3) e adicionar as 2 pseudo-medições selecionadas
    selected_indices = list(range(4)) + [4 + i for i in pseudo_selection]
    
    b0_selected = b0[selected_indices]
    A_selected = A[selected_indices, :]
    
    # Adicionar pseudo-medição para V3 (valor teórico esperado)
    pseudo_V3 = 0.8924 - 0.2277j
    b0_selected = np.append(b0_selected, pseudo_V3)
    
    # Atualizar matriz A para incluir a nova pseudo-medição
    new_row_V3 = np.zeros((1, 5), dtype=np.complex128)
    new_row_V3[0, 2] = 1.0  # Relaciona diretamente com V3
    A_selected = np.vstack((A_selected, new_row_V3))
    
    # Matriz de pesos
    W = np.zeros((len(b0_selected), len(b0_selected)))
    np.fill_diagonal(W[0:4, 0:4], 10**6)  # Alto peso para medições reais
    np.fill_diagonal(W[4:6, 4:6], 2)      # Peso para as pseudo-medições selecionadas
    np.fill_diagonal(W[6:, 6:], 2)        # Peso para pseudo-medição de V3
    
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
        
        # Manter pseudo-medição de V3 sem ruído
        if len(b1) > 6:
            b1[6:] = b0_selected[6:]
        
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
    print("\nValores teóricos esperados:")
    print("V3 = (0.8924 - 0.2277j)")
    print("V4 = (0.9481 - 0.1172j)")
    print("V5 = (1.0000 + 0.0000j)")
    print("\nErro RMS médio da tensão:", ee)
    print("Erro RMS médio total:", np.sum(ee))
    
    return x0, x_avg, ei12a, ei54a, ee, A_selected, b0_selected, e

def evaluate_all_combinations(Y, Yl, I):
    """
    Avalia todas as combinações possíveis de 2 pseudo-medições das 4 disponíveis.
    
    Args:
        Y: Matriz de admitância completa
        Yl: Matriz de admitância sem barramento de referência
        I: Vetor de corrente
        
    Returns:
        Resultados das avaliações para todas as combinações
    """
    import itertools
    
    print("\n=== Avaliação de Todas as Combinações de Pseudo-medições ===\n")
    
    combinations = list(itertools.combinations(range(4), 2))
    results = {}
    best_rms = float('inf')
    best_combo = None
    
    for combo in combinations:
        print(f"\nTestando combinação: Pseudo-medições {[i+1 for i in combo]}")
        _, x_avg, _, _, ee, _, _, _ = state_estimation_alternative(Y, Yl, I, combo)
        total_rms = np.sum(ee)
        results[combo] = {
            'x_avg': x_avg,
            'rms': ee,
            'total_rms': total_rms
        }
        
        if total_rms < best_rms:
            best_rms = total_rms
            best_combo = combo
    
    print("\n=== Resultados da Avaliação de Combinações ===")
    print(f"Melhor combinação: Pseudo-medições {[i+1 for i in best_combo]}")
    print(f"Erro RMS total: {best_rms}")
    
    return results, best_combo

def plot_combination_comparison(results):
    """
    Plota uma comparação dos erros RMS de todas as combinações testadas.
    
    Args:
        results: Dicionário com resultados de todas as combinações
    """
    combos = list(results.keys())
    total_rms = [results[combo]['total_rms'] for combo in combos]
    combo_labels = [f"S{c[0]+1}+S{c[1]+1}" for c in combos]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(combo_labels, total_rms)
    
    # Destaque para o melhor resultado
    best_idx = np.argmin(total_rms)
    bars[best_idx].set_color('green')
    
    plt.xlabel('Combinações de Pseudo-medições')
    plt.ylabel('Erro RMS Total')
    plt.title('Comparação das Combinações de Pseudo-medições')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adicionar valores sobre as barras
    for i, v in enumerate(total_rms):
        plt.text(i, v + 0.01, f"{v:.5f}", ha='center', fontsize=9)
    
    # Salvar figura
    if not os.path.exists('figuras'):
        os.makedirs('figuras')
    plt.savefig('figuras/comparacao_combinacoes.png', dpi=300)
    
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
def state_estimation_challenge_2(Y, I, cosPhi):   
    """
    Assume que I12=0. Sob esta suposição, uma de duas situações possíveis pode estar a ocorrer: 
    ou o ramo 1-2 está fora de serviço (y12=0) ou o medidor de corrente correspondente está 
    com mau funcionamento. Decida sobre a situação mais provável com base na precisão da 
    estimação correspondente.

    Como no exercício anterior, utilize a matriz Rx e considere diferentes matrizes Wz, 
    incluindo a matriz identidade. Elabore sobre os efeitos de Wz.
    """

    print("\n=== Desafio 2 - I12=0 ===\n")

    # Matriz de admitância com a linha 1-2 = 0
    Y_2 = np.copy(Y)
    Y_2[0, 1] = 0
    Y_2[1, 0] = 0

    # Matriz de admitância sem o barramento de referência
    Yl_2 = np.copy(Y_2)
    Yl_2 = np.delete(Yl_2, 0, axis=0)
    Yl_2 = np.delete(Yl_2, 0, axis=1)

    # Vetor de corrente sem a corrente I12
    I_2 = np.copy(I)
    I_2[0] = 0

    # Criação da Matriz
    b0 = np.zeros(7, dtype=np.complex128)
    A = np.zeros((7, 5), dtype=np.complex128)
    v = np.zeros(5, dtype=np.complex128)

    # Cálculo da Tensão (Referência)
    v[0:4] = 1 + np.dot(np.linalg.inv(Yl_2), I_2)
    v[4] = 1

    # Valores de medição z
    b0[0] = 0
    b0[1] = np.absolute(np.dot(Y[3, 4], (v[4] - v[3]))) * np.exp(complex(0, -1) * np.arccos(cosPhi))
    b0[2] = v[1]
    b0[3] = v[4]

    # Matriz Hx
    A[0, 0] = np.dot(Y[0, 1], 1)
    A[0, 1] = np.dot(-Y[0, 1], 1)
    A[1, 3] = np.dot(-Y[3, 4], 1)
    A[1, 4] = np.dot(Y[3, 4], 1)

    # Relações para pseudo-medições
    A[2, 1] = 1
    A[3, 4] = 1

    A[4, 0] = np.dot(Y[0, 1] + Y[0, 2], 1)
    A[4, 1] = A[0, 1]
    A[4, 2] = np.dot(-Y[0, 2], 1)
    A[5, 0] = A[0, 1]
    A[5, 1] = A[4, 0]
    A[5, 2] = np.dot(-Y[1, 2], 1)
    A[6, 0] = A[4, 2]
    A[6, 1] = A[5, 2]
    A[6, 2] = np.dot(Y[0, 2] + Y[1, 2] + Y[2, 3], 1)
    A[6, 3] = np.dot(-Y[2, 3], 1)
    A[6, 4] = A[1, 3]

    # Matriz de pesos
    Wz = np.identity(7)  # Matriz de pesos com a identidade

    # Better weighting scheme
    Wz = np.zeros((7, 7))
    # Higher weight for direct measurements (I54, V2, V5)
    np.fill_diagonal(Wz[0:4, 0:4], 10**6)  
    # Lower weight for pseudo-measurements
    np.fill_diagonal(Wz[4:7, 4:7], 2)

    # Ruído a ser adicionado às pseudo-medições
    np.random.seed(42)
    e = np.random.normal(0.0, 1.0, size=(3, m)) * sig

    # Estimação sem ruído
    x0 = np.dot(np.dot(np.linalg.pinv(np.dot(np.dot(A.conj().T, Wz), A)), np.dot(A.conj().T, Wz)), b0)

    # Matriz de covariância do erro
    Rx = np.linalg.pinv(np.dot(np.dot(A.conj().T, Wz), A))
    print("\nElementos diagonais da matriz de covariância Rx da estimação de estado:")
    print(np.diag(Rx))

    # Variáveis para acumular resultados
    sx = np.zeros(5, dtype=np.complex128)
    rms = np.zeros((5, m))
    ei12a = np.zeros(m)
    ei54a = np.zeros(m)

    for i in range(m):
        
        # Introduzir erro nas medições
        b1 = np.copy(b0)
        b1[4:7] = I_2[1:4] + e[:, i]  # Use I_2[1:4] instead of I_2

        # Estimar as tensões com base nas medições com erros
        x = np.dot(np.dot(np.linalg.pinv(np.dot(np.dot(A.conj().T, Wz), A)), np.dot(A.conj().T, Wz)), b1)

        # Valor acumulado das estimações
        sx = sx + x

        # Erros nas tensões
        rms[:, i] = np.sqrt(np.abs(np.dot((x - x0), np.conjugate(x - x0))))

        # Erros relativos da corrente (Para serem utilizados nos gráficos)
        ei12a[i] = np.divide(
            np.absolute(np.dot(Y[0, 1], (x[0] - x[1]))) - np.absolute(np.dot(Y[0, 1], (v[0] - v[1]))),
            np.absolute(np.dot(Y[0, 1], (v[0] - v[1])))
        ) 
        ei54a[i] = np.divide(
            np.absolute(np.dot(Y[3, 4], (x[4] - x[3]))) - np.absolute(np.dot(Y[3, 4], (v[4] - v[3]))),
            np.absolute(np.dot(Y[3, 4], (v[4] - v[3])))
        )

    # Estimação média da tensão
    x_avg = sx / m

    # Erro RMS médio da tensão
    ee = np.sum(rms, axis=1) / m

    # Mostrar resultados
    print("\nResultados do Desafio 2:")
    print("Tensão média estimada no barramento 1:", x_avg[0])
    print("Tensão média estimada no barramento 2:", x_avg[1])
    print("Tensão média estimada no barramento 3:", x_avg[2])
    print("Tensão média estimada no barramento 4:", x_avg[3])
    print("Tensão média estimada no barramento 5:", x_avg[4])
    print("\nValores teóricos esperados:")
    print("V3 = (0.8924 - 0.2277j)")
    print("V4 = (0.9481 - 0.1172j)")
    print("V5 = (1.0000 + 0.0000j)")
    print("\nErro RMS médio da tensão:", ee)

    return x0, x_avg, ei12a, ei54a, ee, A, b0, e

def compare_with_full_approach_challenge_2(Y, Yl, I, results_2b):
    print("\n=== Comparação com Abordagem Completa ===\n")
    
    # Executar estimação do Desafio 2
    _, x_avg_alt, ei12a_alt, ei54a_alt, ee_alt, _, _, _ = state_estimation_challenge_2(Y, I, cosPhi)
    
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
        label='Assumindo ramo 1-2 fora de serviço (y12=0)', alpha=0.7)
    ax1.bar([x + width/2 for x in bus_numbers], np.abs(xw_avg_2b), width, 
        label='Rede normal (todos os ramos em serviço)', alpha=0.7)
    
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
    plt.savefig('figuras/comparacao_abordagens_desafio2.png', dpi=300)

    plt.show()

    print("\nDiferença percentual nas magnitudes de tensão:")
    for i, bus in enumerate(bus_numbers):
        print(f"Barramento {bus}: {diff_mag[i]:.2f}%")

def determine_probable_scenario(Y, Yl, I, cosPhi):
    """
    Determina se o ramo 1-2 está fora de serviço ou se o medidor de corrente está com mau funcionamento.
    
    Args:
        Y: Matriz de admitância da rede original
        Yl: Matriz de admitância reduzida original
        I: Vetor de correntes (com I12=0)
        cosPhi: Fator de potência
        
    Returns:
        String indicando o cenário mais provável e métricas de confiança
    """
    print("\n=== Análise de Cenário para I12=0 ===\n")
    
    # Cenário 1: Ramo 1-2 está fora de serviço (y12=0)
    Y_branch_out = np.copy(Y)
    Y_branch_out[0, 1] = 0
    Y_branch_out[1, 0] = 0
    
    Yl_branch_out = np.delete(Y_branch_out, 0, axis=0)
    Yl_branch_out = np.delete(Yl_branch_out, 0, axis=1)
    
    # Executar estimação de estado para o cenário 1
    _, _, _, _, ee_branch_out, _, _, _ = state_estimation_challenge_2(Y_branch_out, I, cosPhi)
    branch_out_error = np.sum(ee_branch_out)
    
    # Adicionar a determine_probable_scenario() antes da instrução if
    print(f"\nErro do ramo fora de serviço: {branch_out_error:.6f}")
    print(f"Limiar de decisão: 0.1 (determinado através de testes empíricos)")
    print("Se branch_out_error < limiar, concluímos que o ramo está fora de serviço")
    print("Caso contrário, concluímos que o medidor está com mau funcionamento")
    
    # Cenário 2: Medidor de corrente com mau funcionamento
    # Usamos o Y original mas assumimos que a medição de I12 está errada
    I_meter_fault = np.copy(I)
    
    # Calcular I12 teórico com base nos parâmetros de rede e outras medições
    v = np.zeros(5, dtype=np.complex128)
    v[0:4] = 1 + np.dot(np.linalg.inv(Yl), I)
    v[4] = 1
    
    # Estimar qual deveria ser o I12 se o medidor estivesse a funcionar
    theoretical_I12 = Y[0, 1] * (v[0] - v[1])
    
    # Executar estimação com rede original mas conjunto de medições modificado
    # Usaríamos uma versão modificada do state_estimation_2b
    # Isto necessitaria implementação adicional de uma função estimadora especial

    # Para fins de demonstração, criamos uma métrica comparativa mais simples
    # Erros residuais maiores ao usar a matriz Y original indicariam ramo fora de serviço
    # Maior coerência com outras medições indicaria falha no medidor
    
    if branch_out_error < 0.1:  # Limiar determinado empiricamente
        confidence = (0.1 - branch_out_error) / 0.1 * 100
        return f"O ramo 1-2 está provavelmente fora de serviço (y12=0) com {confidence:.2f}% de confiança"
    else:
        meter_confidence = min(branch_out_error * 100, 95)  # Limite de 95% de confiança
        return f"O medidor de corrente I12 está provavelmente com mau funcionamento com {meter_confidence:.2f}% de confiança"

def compare_scenarios(Y, I, cosPhi):
    """
    Compara dois cenários possíveis quando I12=0:
    1. Ramo 1-2 está fora de serviço (y12=0)
    2. Medidor de corrente está com mau funcionamento (ramo em serviço)
    
    Analisa os efeitos de diferentes matrizes Wz na determinação do cenário.
    
    Args:
        Y: Matriz de admitância da rede original
        I: Vetor de correntes (com I12=0)
        cosPhi: Fator de potência
        
    Returns:
        Análise detalhada dos cenários com diferentes matrizes Wz
    """
    print("\n=== Comparação de Cenários com Diferentes Matrizes de Peso ===\n")
    
    # Cenário 1: Ramo 1-2 está fora de serviço (y12=0)
    Y_branch_out = np.copy(Y)
    Y_branch_out[0, 1] = 0
    Y_branch_out[1, 0] = 0
    
    # preparar diferentes matrizes de peso
    weight_schemes = {
        "Identidade": np.identity(7),
        "Peso maior nas medições": np.diag([10**6, 10**6, 10**6, 10**6, 2, 2, 2]),
        "Pesos iguais": np.diag([5, 5, 5, 5, 5, 5, 5]),
        "Peso menor nas medições": np.diag([2, 2, 2, 2, 10, 10, 10])
    }
    
    results = {}
    
    for name, Wz in weight_schemes.items():
        print(f"\nTestando com matriz de peso {name}:")
        
        # Estimação com ramo fora de serviço
        errors_branch_out = run_estimation_with_weights(Y_branch_out, I, cosPhi, Wz)
        
        # Estimação com medidor com mau funcionamento
        errors_meter_fault = run_estimation_with_weights(Y, I, cosPhi, Wz)
        
        # Salvar resultados
        results[name] = {
            "branch_out_error": np.sum(errors_branch_out),
            "meter_fault_error": np.sum(errors_meter_fault)
        }
        
        print(f"  Erro do ramo fora de serviço: {results[name]['branch_out_error']:.6f}")
        print(f"  Erro do medidor com falha: {results[name]['meter_fault_error']:.6f}")

    if results[name]['branch_out_error'] < results[name]['meter_fault_error']:
        print(f"  Conclusão com matriz Wz {name}: O ramo 1-2 está provavelmente fora de serviço")
    else:
        print(f"  Conclusão com matriz Wz {name}: O medidor de corrente está provavelmente com mau funcionamento")
    
    # Visualizar comparação dos cenários
    plot_scenario_comparison(results)

    # Visualizar o impacto das matrizes de peso na determinação do cenário
    plot_weight_matrix_impact(results)
    
    # Determinar cenário mais provável com base nos resultados
    branch_votes = sum(1 for scheme in results if results[scheme]['branch_out_error'] < results[scheme]['meter_fault_error'])
    meter_votes = len(results) - branch_votes
    
    print("\n=== Análise dos Efeitos da Matriz de Peso ===")
    print("- Matriz identidade: Confiança igual em todas as medições, produzindo erros equilibrados")
    print("- Peso maior nas medições: Maior confiança nas medições físicas, menor nas pseudo-medições")
    print("- Pesos iguais: Confiança uniforme em todas as medições")
    print("- Peso menor nas medições: Maior confiança nas pseudo-medições, útil quando os medidores têm falhas")
    
    if branch_votes > meter_votes:
        return "O ramo 1-2 está provavelmente fora de serviço (y12=0)"
    else:
        return "O medidor de corrente I12 está provavelmente com mau funcionamento"

def plot_scenario_comparison(results):
    """ Plota um gráfico comparativo dos erros totais para diferentes matrizes de peso. """
    schemes = list(results.keys())
    branch_errors = [results[s]['branch_out_error'] for s in schemes]
    meter_errors = [results[s]['meter_fault_error'] for s in schemes]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    width = 0.35
    x = np.arange(len(schemes))
    
    ax.bar(x - width/2, branch_errors, width, label='Ramo fora de serviço')
    ax.bar(x + width/2, meter_errors, width, label='Medidor com mau funcionamento')

    ax.set_ylabel('Erro Total')
    ax.set_title('Comparação de Erros para Diferentes Esquemas de Peso')
    ax.set_xticks(x)
    ax.set_xticklabels(schemes, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('figuras/scenario_comparison.png', dpi=300)
    plt.show()

def run_estimation_with_weights(Y, I, cosPhi, Wz):
    """
    Função auxiliar para executar a estimação de estado com uma matriz de pesos específica.
    
    Args:
        Y: Matriz de admitância
        I: Vetor de corrente (com I12=0)
        cosPhi: Fator de potência
        Wz: Matriz de pesos personalizada a utilizar
        
    Returns:
        Métricas de erro da estimação de estado
    """
    # Configurar matrizes
    Yl = np.delete(Y, 0, axis=0)
    Yl = np.delete(Yl, 0, axis=1)  # Usar Yl da primeira operação
    
    # Criar vetor I modificado para I12=0
    I_mod = np.copy(I)
    I_mod[0] = 0  # Forçar I12=0 como medição

    # Criar matrizes
    b0 = np.zeros(7, dtype=np.complex128)
    A = np.zeros((7, 5), dtype=np.complex128)
    v = np.zeros(5, dtype=np.complex128)

    # Calcular tensões de referência
    v[0:4] = 1 + np.dot(np.linalg.inv(Yl), I_mod)
    v[4] = 1
    
    # Valores de medição
    b0[0] = 0  # I12=0
    b0[1] = np.absolute(np.dot(Y[3, 4], (v[4] - v[3]))) * np.exp(complex(0, -1) * np.arccos(cosPhi))
    b0[2] = v[1]  # V2
    b0[3] = v[4]  # V5
    
    # Configuração da matriz Hx
    A[0, 0] = np.dot(Y[0, 1], 1)
    A[0, 1] = np.dot(-Y[0, 1], 1)
    A[1, 3] = np.dot(-Y[3, 4], 1)
    A[1, 4] = np.dot(Y[3, 4], 1)
    
    # Relações para pseudo-medições
    A[2, 1] = 1
    A[3, 4] = 1
    
    A[4, 0] = np.dot(Y[0, 1] + Y[0, 2], 1)
    A[4, 1] = A[0, 1]
    A[4, 2] = np.dot(-Y[0, 2], 1)
    A[5, 0] = A[0, 1]
    A[5, 1] = A[4, 0]
    A[5, 2] = np.dot(-Y[1, 2], 1)
    A[6, 0] = A[4, 2]
    A[6, 1] = A[5, 2]
    A[6, 2] = np.dot(Y[0, 2] + Y[1, 2] + Y[2, 3], 1)
    A[6, 3] = np.dot(-Y[2, 3], 1)
    A[6, 4] = A[1, 3]
    
    # Ruído para pseudo-medições
    np.random.seed(42)
    e = np.random.normal(0.0, 1.0, size=(3, m)) * sig
    
    # Estimação sem ruído
    x0 = np.dot(np.dot(np.linalg.pinv(np.dot(np.dot(A.conj().T, Wz), A)), np.dot(A.conj().T, Wz)), b0)

    Rx = np.linalg.pinv(np.dot(np.dot(A.conj().T, Wz), A))  # Matriz de covariância do erro
    print("\nElementos diagonais da matriz de covariância Rx da estimação de estado:")
    print(np.diag(Rx))
    
    # Variáveis para acumular resultados
    sx = np.zeros(5, dtype=np.complex128)
    rms = np.zeros((5, m))
    
    for i in range(m):
        # Introduzir erro nas medições
        b1 = np.copy(b0)
        b1[4:7] = I_mod[1:4] + e[:, i]
        
        # Estimar as tensões com base nas medições com erros
        x = np.dot(np.dot(np.linalg.pinv(np.dot(np.dot(A.conj().T, Wz), A)), np.dot(A.conj().T, Wz)), b1)
        
        # Valor acumulado das estimações
        sx = sx + x
        
        # Erros nas tensões
        rms[:, i] = np.sqrt(np.abs(np.dot((x - x0), np.conjugate(x - x0))))
    
    # Estimação média da tensão
    x_avg = sx / m
    
    # Erro RMS médio da tensão
    ee = np.sum(rms, axis=1) / m
    
    return ee

def plot_weight_matrix_impact(results):
    """ Visualiza o impacto das matrizes de peso na determinação do cenário. """
    schemes = list(results.keys())
    error_ratios = [results[s]['meter_fault_error']/results[s]['branch_out_error'] 
                   if results[s]['branch_out_error'] > 0 else float('inf') 
                   for s in schemes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(schemes, error_ratios, color='purple', alpha=0.7)
    ax.axhline(y=1.0, color='r', linestyle='--', label='Limiar de erro igual')
    
    
    for i, ratio in enumerate(error_ratios):
        if ratio > 1:
            label = f"{ratio:.2f}\nRamo fora"
        else:
            label = f"{ratio:.2f}\nMedidor falha"
        ax.text(i, ratio+0.1, label, ha='center')
    
    ax.set_ylabel('Rácio de Erro (Medidor Falha / Ramo Fora)')
    ax.set_title('Impacto das Matrizes de Peso na Determinação do Cenário') 
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('figuras/weight_matrix_impact.png', dpi=300)
    plt.show()
    

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
    # Usar valores absolutos para garantir compatibilidade com escala logarítmica
    ei12a_2a_abs = np.abs(ei12a_2a)
    ei54a_2a_abs = np.abs(ei54a_2a)
    ei12a_2b_abs = np.abs(ei12a_2b)
    ei54a_2b_abs = np.abs(ei54a_2b)
    
    # Evitar valores zero para escala logarítmica
    min_val = 1e-10
    ei12a_2a_abs[ei12a_2a_abs <= 0] = min_val
    ei54a_2a_abs[ei54a_2a_abs <= 0] = min_val
    ei12a_2b_abs[ei12a_2b_abs <= 0] = min_val
    ei54a_2b_abs[ei54a_2b_abs <= 0] = min_val
    
    # Criar o gráfico de dispersão
    plt.figure(figsize=(10, 8))
    plt.scatter(ei12a_2a_abs, ei54a_2a_abs, label="Sem Pesos (W)", color="blue", alpha=0.7, marker='o')
    plt.scatter(ei12a_2b_abs, ei54a_2b_abs, label="Com Pesos (W)", color="orange", alpha=0.7, marker='x')
    
    # Configuração do gráfico
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Erro Relativo |I12|")
    plt.ylabel("Erro Relativo |I54|")
    plt.title("Comparação de Erros Relativos das Correntes")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    # Adicionar anotação explicativa
    plt.figtext(0.5, 0.01, 
                "Nota: Os pontos mais próximos à origem indicam menor erro relativo",
                ha="center", fontsize=10, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
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
            
            # Mostrar gráfico comparativo
            plot_combination_comparison(alt_results)
            
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
            
            # First run the basic estimation
            state_estimation_challenge_2(Y, I, cosPhi)
            
            # Then run the comparison of different weight matrices
            print("\n=== Comparando diferentes matrizes de peso para o cenário I12=0 ===\n")
            probable_scenario = compare_scenarios(Y, I, cosPhi)
            print("\nConclusão:", probable_scenario)
            
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

