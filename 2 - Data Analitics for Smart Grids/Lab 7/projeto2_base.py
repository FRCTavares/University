###############################################################################################
# Projeto 2 - Estimação das Correntes de Carga e das Impedâncias das Linhas numa Rede de BT   #
#                                                                                             #
# Grupo 13                                                                                    #
#                                                                                             #
# Membros:                                                                                    #
#   Francisco Tavares (103402)                                                                #
#   Marta Valente (103574)                                                                    #
###############################################################################################

# =============================================================================================
# Importação de bibliotecas
# =============================================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================================
# Parâmetros globais
# =============================================================================================
cosPhi = 0.95
m = 12                # Número de períodos (ex.: 12 períodos de 15 min → 3 horas)
netFactor = 0.25      # Fator da rede

# =============================================================================================
# Dados 
# =============================================================================================
def load_data():
    # Consumption dataset: cada linha corresponde a uma medição (a cada 15 minutos)
    s = [[0.0450, 0.0150, 0.0470, 0.0330],
         [0.0250, 0.0150, 0.2480, 0.0330],
         [0.0970, 0.0250, 0.3940, 0.0330],
         [0.0700, 0.0490, 0.0200, 0.4850],
         [0.1250, 0.0460, 0.0160, 0.1430],
         [0.2900, 0.0270, 0.0160, 0.0470],
         [0.2590, 0.0150, 0.0170, 0.0200],
         [0.2590, 0.0160, 0.0280, 0.0160],
         [0.4420, 0.0160, 0.0500, 0.0170],
         [0.2010, 0.0230, 0.0460, 0.0160],
         [0.2060, 0.0490, 0.0220, 0.0240],
         [0.1300, 0.0470, 0.0160, 0.0490],
         [0.0460, 0.0260, 0.0170, 0.0480]]
    s = np.array(s)

    # Topologia da rede: rede radial com 4 nós
    topo = [[1, 2], [2, 3], [3, 4]]
    nBUS = np.max(topo)

    # Impedâncias das linhas (já escaladas pelo netFactor)
    z = np.multiply([complex(0.1, 0.05), complex(0.15, 0.07), complex(0.2, 0.1)], netFactor)

    vr = 1    # Tensão de referência
    el = 1    # Expoente do cálculo de tensão (usado na fórmula de potência)
    ni = 20   # Número de iterações para o cálculo do Power Flow

    return s, topo, nBUS, z, vr, el, ni

# =============================================================================================
# Função de Power Flow
# =============================================================================================
def pf3ph(t, z, si, vr, el, ni, al, nBUS):
    """
    Cálculo do fluxo de potência trifásico para uma rede radial.
    
    Args:
        t: topologia da rede (lista de pares [pai, filho])
        z: vetor de impedâncias de cada linha
        si: matriz de potência injetada em cada fase e em cada bus (3x4)
        vr: tensão de referência
        el: expoente de tensão
        ni: número de iterações
        al: ângulo de fase (usualmente exp(-j*2π/3))
        nBUS: número de barras (nós)
        
    Returns:
        mvp: matriz de tensões de fase em cada nó (3 x número de nós)
        ip: matriz de correntes de fase (3 x número de nós)
    """
    t = np.array(t)
    p = t[:, 0]
    f = t[:, 1]
    w = len(p) + 1  # número de nós = número de ramos + 1
    
    vp = np.zeros((nBUS - 1, w), dtype=complex)
    vn = np.zeros((nBUS - 1, w), dtype=complex)
    
    # Inicializa a tensão de referência na primeira barra
    vp[0, 0:w] = vr

    # Propaga a tensão pelos nós usando a rotação (assumindo tensões iguais em todas as barras)
    for h in range(2, nBUS):
        vp[h - 1, :] = vp[h - 2, :] * al

    va = vp - vn  # Tensão auxiliar
    ia = np.conjugate(np.divide(np.multiply(si, np.abs(va)**el), va))
    
    for it_iter in range(ni):
        va = vp - vn
        ip = np.conjugate(np.divide(np.multiply(si, np.abs(va)**el), va))
        inn = -np.sum(ip, 0)  # Corrente do neutro

        for k in range(w - 1, 0, -1):  # Ciclo Backward
            n_ = f[k - 1]
            m_ = p[k - 1]
            ip[:, m_ - 1] = ip[:, m_ - 1] + ip[:, n_ - 1]
            inn = -np.sum(ip, 0)

        eps = np.linalg.norm(np.max(np.abs(ia - ip), 0))
        if eps <= 1e-4:
            mvp = vp - vn
            return mvp, ip
        else:
            ia = ip

        for k in range(w - 1):  # Ciclo Forward
            n_ = f[k]
            m_ = p[k]
            vn[:, n_ - 1] = vn[:, m_ - 1] - z[k] * inn[n_ - 1]
            vp[:, n_ - 1] = vp[:, m_ - 1] - z[k] * ip[:, n_ - 1]
        ia = ip

    # Caso não converja, retorna as últimas estimativas
    return (vp - vn), ip

# =============================================================================================
# Função de Estimativa de Correntes com Regularização
# =============================================================================================
def base_estimation():
    """
    A função calcula a estimativa de corrente utilizando o método de mínimos quadrados.
    """
    # Carregar dados
    s, topo, nBUS, z, vr, el, ni = load_data()

    al=np.exp(np.multiply(np.multiply(complex(0,-1),2/3),np.pi)) #Phase Angle
    sp=np.mean(s[0:m,:], axis=0) #Average power in each phase (i0)

    si=[[0, 0, sp[2], 0],[0, 0, sp[1], 0],[0, sp[0],  0, sp[3]]] #Power in each bus and in each phase

    mvp, ip=pf3ph(topo, z, si, vr, el, ni, al, nBUS)  #Compute the power flow

    zv=mvp[:,3]    #Voltage Measurement in Node 4 

    scale=1e-9*np.abs(1/sp)**2  #Measurements Accuracy

    #Matrices Creation
    it=np.zeros((3,m))
    ie=np.zeros((3,m))
    dx = np.zeros((4,m))
    A=np.zeros((3,4), dtype=complex)


#######################################################################################################
    # Matriz A
  
    # Gamma matrix for the 4 phases
    gamma1 = np.array([0, 0, al**2])
    gamma2 = np.array([0, al, 0])
    gamma3 = np.array([1, 0, 0])

    W = np.array([[2,1,1], [1,2,1], [1,1,2]])
    
    z_accumulated = [
    z[0],                 # Node 1: z[0] (impedância entre source e nó 1)
    z[0] + z[1],          # Node 2: z[0] + z[1] (impedância acumulada até nó 2)
    z[0] + z[1],          # Node 3: z[0] + z[1] (impedância acumulada até nó 3, igual ao nó 2)
    z[0] + z[1] + z[2]    # Node 4: z[0] + z[1] + z[2] (impedância acumulada até nó 4)
    ]  

    Wk1 = np.zeros((3, 3), dtype=complex)
    Wk2 = np.zeros((3, 3), dtype=complex)
    Wk3 = np.zeros((3, 3), dtype=complex)
    Wk4 = np.zeros((3, 3), dtype=complex)

    Wk1 = z_accumulated[0] * W
    Wk2 = z_accumulated[1] * W
    Wk3 = z_accumulated[2] * W
    Wk4 = z_accumulated[3] * W

    #Compute matrix A
    # Phase a (row 0)
    A[0,0] = np.dot(Wk1[0], gamma1)
    A[0,1] = np.dot(Wk2[0], gamma2)
    A[0,2] = np.dot(Wk3[0], gamma3)
    A[0,3] = np.dot(Wk4[0], gamma1)

    # Phase b (row 1)
    A[1,0] = np.dot(Wk1[1], gamma1)
    A[1,1] = np.dot(Wk2[1], gamma2)
    A[1,2] = np.dot(Wk3[1], gamma3)
    A[1,3] = np.dot(Wk4[1], gamma1)

    # Phase c (row 2)
    A[2,0] = np.dot(Wk1[2], gamma1)
    A[2,1] = np.dot(Wk2[2], gamma2)
    A[2,2] = np.dot(Wk3[2], gamma3)
    A[2,3] = np.dot(Wk4[2], gamma1)

#######################################################################################################
    for i in range(m):
        #Power in each instant 
        si=np.zeros(shape=[3,4]) # Three lines one for each phase and four buses
        si[2,1]=s[i,0]
        si[0,2]=s[i,1]
        si[1,2]=s[i,2]
        si[2,3]=s[i,3]
        # Power Flow
        mvp, ip = pf3ph(topo, z, si, vr, el, ni, al, nBUS) # Power Flow
        v4 = mvp[:,3] # Voltage at bus 4
        dv=v4-zv # Voltage difference

        Atrans=np.transpose(A)
        di=np.dot(np.dot(np.linalg.inv(np.dot(Atrans,A)+np.diag(scale)), Atrans),-dv) #Min-norm solution

        dx[:,i]=di # Store the result in the matrix dx


#####################################################################################################################################



    return dx # Return the matrix with the results of the current estimation

  

# =============================================================================================
# Função do Menu e Principal
# =============================================================================================
def show_menu():
    """Exibe o menu principal e retorna a opção selecionada pelo usuário."""
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 80)
    print("                             Projeto 2 - Grupo 13")
    print("=" * 80)
    print("\nEscolha uma opção (deve ser por ordem):")
    print("0 - Sair")
    print("1 - Rede LV")
    print("=" * 80)
    try:
        option_input = input("\nOpção: ").strip()
        show_plots = False
        if option_input.endswith('p'):
            show_plots = True
            option_input = option_input[:-1]
        option = int(option_input)
        return option, show_plots
    except ValueError:
        return -1, False

def main():
    # Carregar dados
    s, topo, nBUS, z, vr, el, ni = load_data()
    # Conexões de fase não são necessárias explicitamente, pois já estão codificadas em gamma
    # Calcular pseudo-medições a partir dos m primeiros instantes (média de s)
    sp = np.mean(s[0:m, :], axis=0)  # vetor de tamanho 4
    # Construir potência média em cada fase (si) com base na média sp
    si = [[0, 0, sp[2], 0],
          [0, 0, sp[1], 0],
          [0, sp[0], 0, sp[3]]]
    
    while True:
        option, show_plots = show_menu()
        if option == 0:
            break
        elif option == 1:
            print("Rede LV selecionada.")
            # Estimar as variações de corrente usando regularização
            di = base_estimation()
            print("\nDelta i (variação de corrente estimada):")
            print(di)
            input("\nPressione Enter para continuar...")
        else:
            print("Opção inválida. Tente novamente.")
            input("Pressione Enter para continuar...")

if __name__ == "__main__":
    main()
