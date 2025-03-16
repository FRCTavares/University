###############################################################################################
# Laboratório 4 - Regressão sobre Previsão Autoregressiva                                     #
#                                                                                             #
# Grupo 13                                                                                    #
#                                                                                             #
# Membros:                                                                                    #
#   Francisco Tavares (103402)                                                                #
#   Marta Valente (103574)                                                                    #
###############################################################################################

# ============================================================================================================================================
# Importação de bibliotecas
# ============================================================================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================================================================================================
# Parâmetros Globais
# ============================================================================================================================================
networkFactor = 100    # Para alterar as características da rede (Y)
cosPhi=0.95            # Valor de teta
time=24                # Período de Treino    
timeForecast=12        # Período de Teste

# ============================================================================================================================================
# Funções para carregar e processar dados
# ============================================================================================================================================
def load_data(file_path):
    Info = np.array(pd.read_excel (r'DASG_Prob2_new.xlsx', sheet_name='Info', header=None))
    # Informação sobre o barramento de referênciae referência
    SlackBus=Info[0,1]
    print ("Barramento de referência: ", SlackBus)

    # Informação da Rede
    Net_Info = np.array(pd.read_excel (r'DASG_Prob2_new.xlsx', sheet_name='Y_Data'))
    print ("Informação das linhas (Admitâncias)\n", Net_Info, "\n")

    #Informação de Potência (Treino)reino)
    Power_Info = np.array(pd.read_excel (r'DASG_Prob2_new.xlsx', sheet_name='Load(t,Bus)'))
    Power_Info = np.delete(Power_Info,[0],1)
    print ("Informação do consumo de potência (tempo, Barramento)\n", Power_Info, "\n")

    #Informação de Potência (Teste)Teste)
    Power_Test = np.array(pd.read_excel (r'DASG_Prob2_new.xlsx', sheet_name='Test_Load(t,Bus)'))
    Power_Test = np.delete(Power_Test,[0],1)
    print ("Informação do consumo de potência (tempo, Barramento)\n", Power_Test, "\n")

    return SlackBus, Net_Info, Power_Info, Power_Test

def prepare_network(Net_Info, Power_Info, SlackBus):
    P=np.dot(-Power_Info,np.exp(complex(0,1)*np.arccos(cosPhi)))
    I=np.conj(P[2,:])


    P=np.dot(-Power_Info,np.exp(complex(0,1)*np.arccos(cosPhi)))
    print(P)
    I=np.conj(P[2,:])


    # Determinar o número de Barramentosamentos
    nBus=max(np.max(Net_Info[:,0]),np.max(Net_Info[:,1]))

    # Criar a variável número de linhas e a matriz de admitâncias (Y)
    nLines=Net_Info.shape[0]

    Y=np.zeros((nBus,nBus), dtype=complex)

    #Completar a matriz Y e atualizar o número de linhas
    for i in range (Net_Info.shape[0]):
        y_aux=Net_Info[i,2].replace(",",".")
        y_aux=y_aux.replace("i","j")
        Y[Net_Info[i,0]-1,Net_Info[i,0]-1]=Y[Net_Info[i,0]-1,Net_Info[i,0]-1]+complex(y_aux)*networkFactor
        Y[Net_Info[i,1]-1,Net_Info[i,1]-1]=Y[Net_Info[i,1]-1,Net_Info[i,1]-1]+complex(y_aux)*networkFactor
        Y[Net_Info[i,0]-1,Net_Info[i,1]-1]=Y[Net_Info[i,0]-1,Net_Info[i,1]-1]-complex(y_aux)*networkFactor
        Y[Net_Info[i,1]-1,Net_Info[i,0]-1]=Y[Net_Info[i,1]-1,Net_Info[i,0]-1]-complex(y_aux)*networkFactor

                
    # Remover o barramento de referência da matriz de admitâncias            s            
    Yl=np.delete(Y, SlackBus-1, axis=0)
    Yl=np.delete(Yl, SlackBus-1, axis=1)

    # Matriz de Condutânciascias
    G=Yl.real

    # Matriz de Susceptânciasncias
    B=Yl.imag 

    print("A matriz de admitâncias Y é:\n", Y, "\n")
    print("A matriz de condutâncias G é\n", G, "\n")
    print("A matriz de susceptâncias B é\n",B, "\n")

    return Y, Yl, G, B, I, nBus

# ============================================================================================================================================
# Funções para definir a matriz de erros
# ============================================================================================================================================
def error_matrix(Y, Yl, I, nBus, time, timeForecast):
    np.random.seed(50)
    e1=np.random.randn(time+timeForecast)*0.5  #Erros associados à Geração Eólica
    e=np.random.randn(time+timeForecast)*0.25  #Erros associados à Injeção de Potência (Consumo) 


    e1 = [  0.2878,   0.0145,   0.5846,  -0.0029,  -0.2718,  -0.1411,
        -0.2058,  -0.1793,  -0.9878,  -0.4926,  -0.1480,   0.7222,
        -0.3123,   0.4541,   0.9474,  -0.1584,   0.4692,   1.0173,
        -0.0503,   0.4684,  -0.3604,   0.4678,   0.3047,  -1.5098,
        -0.5515,  -0.5159,   0.3657,   0.7160,   0.1407,   0.5424,
            0.0409,   0.0450,   0.2365,  -0.3875,   1.4783,  -0.8487]

    e =  [ -0.0106,   0.0133,   0.2226,   0.2332,   0.1600,  -0.0578,
        -0.2293,  -0.2843,  -0.2732,  -0.1203,  -0.1757,  -0.1891,
            0.1541,  -0.0093,  -0.1691,   0.2211,  -0.4515,  -0.1786,
        -0.2031,  -0.3634,  -0.1105,  -0.1413,  -0.5900,  -0.1729,
        -0.0810,  -0.0023,  -0.0556,   0.1858,  -0.0324,  -0.1071,
        -0.0845,  -0.0743,  -0.0479,  -0.0870,  -0.1834,  -0.1432]


    #Criação da Matriz
    II=np.zeros((nBus-1,time+timeForecast), dtype=complex)
    i12=np.zeros(time+timeForecast)
    i1w=np.zeros(time+timeForecast)
 
    # Inicialização do processo de geração de dados
    II[:,0]=I                                      #Injeções de Potênciancia
    v=1+np.dot(np.linalg.inv(Yl),I)
    i12[0]=np.absolute(np.dot(Y[0,1],v[0]-v[1]))   #Corrente I12 no período t=0
    i1w[0]=np.real(I[0])                           #Injeção no barramento 1 (Eólica) no período t=0

    # Processo de geração de dadosos
    for t in range(time+timeForecast-1):             
        II[:,t+1]=0.95*II[:,t]+e[t]                           # Injeção de potência baseada em períodos anteriores e nos erros..
                                                              # Os valores estão mais ou menos relacionados considerando
                                                              # o valor de 0,95. Este valor pode mudar entre 0 e 1.  
        i1w[t+1]=0.75*i1w[t]+e1[t]                            # Potência eólica baseada nos períodos anteriores  
        II[0,t+1]=i1w[t+1]+np.complex128(0,np.imag(II[0,t+1]))   # Adicionar a geração eólica
        v=1+np.dot(np.linalg.inv(Yl),II[:,t+1])               # Calcular as tensões
        I12=np.dot(-Y[0,1],v[0]-v[1])                         # Calcular o fluxo de carga na linha 1-2 (Complexo)
        i12[t+1]=np.absolute(I12)*np.sign(np.real(I12))       # Calcular o fluxo de carga na linha 1-2 (RMS com sinal)

        
    print ('A injeção de potência no Barramento 1 é:\n',II[0,:])
    print ('\nO fluxo de potência na Linha 1-2 é:\n',i12)

    return II, i12, i1w

# ============================================================================================================================================
# Funções para OLS - FEITO
# ============================================================================================================================================
def OLS(i12, i1w, time, timeForecast):
    # Definir a regressão OLS relacionando a Corrente I12 com a Injeção P1. Ver Equação (30) nas notas da aula
    AA=np.ones((time,2))        
    AA[:,1]=i1w[0:time]         # Vetor Xt com uns na primeira coluna e injeção eólica na coluna 2 
    AATransp=np.transpose(AA)
    beta=np.dot(np.dot(np.linalg.inv(np.dot(AATransp,AA)),AATransp),i12[0:time])  # Valores Beta
    print ("O valor dos Betas, usando OLS, são:\n",beta)

    # Definir os gráficos
    x = range(time)
    yy1 = i12[0:time]
    yy2 = i1w[0:time]
    rss_1=beta[0]+np.dot(beta[1],i1w[0:time])   #Linha de regressão OLS
    yy3 = rss_1
    yy4 = i12[0:time]-beta[0]-np.dot(beta[1],i1w[0:time])

    #Primeiro Gráfico (Injeção de potência no barramento 1 e Corrente I12)to 1 e Corrente I12)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, yy2, 'g-')
    ax2.plot(x, yy1, 'b-')
    ax1.set_xlabel('Instante de Tempo [h]')
    ax1.set_ylabel('Injeção P_1', color='g')
    ax2.set_ylabel('Corrente I_12', color='b')
    ax1.set_ylim([-2, 2])
    ax2.set_ylim([-1, 1])
    plt.xlabel("Instante de Tempo [h]")
    plt.show()

    #Segundo Gráfico (Relação I1 vs I12 e regressão OLS)
    plt.plot(yy2, yy1, 'C1o', label='P1W')
    plt.plot(yy2, yy3, label='Linha de regressão')
    plt.legend()
    plt.xlabel("Injeção P_1")
    plt.ylabel("Corrente I_12")
    plt.show()

    #Terceiro Gráfico (Resíduos - Diferença entre a corrente real I12 e a obtida pela regressão OLS)
    plt.plot(x, yy4, 'C0o', label='Resíduos')
    plt.legend()
    plt.xlabel("Instante de Tempo [h]")
    plt.ylabel("Resíduos")
    plt.show()

    return beta, rss_1

# ============================================================================================================================================
# Funções para Cochrane-Orcutt (CO) - FEITO
# ============================================================================================================================================
def CO(rss_1, beta, b_ss, i12, i1w, time, timeForecast):
    # Definir a regressão Cochrane-Orcutt (CO)
    # Definir os resíduos

    # Cálculo do valor de Durdin-Watson (DW) e do valor de rho
    dw1 = np.sum((rss_1[1:time-1]-rss_1[0:time-2])**2)  # numerador 
    dw2 = np.sum((rss_1[0:time-2])**2)                  # denominador
    D = np.divide(dw1,dw2) 
    ro = 1-D/2
    print ("O valor de Durdin-Watson (DW) é:",D)
    print ("O valor de rho é: ",ro)


    res_1=i12[0:time]-rss_1
    for k in range(3):                  #Iterações para estimar o valor de Beta e Rho
        r2=res_1[0:time-1]
        r1=res_1[1:time]
        ro=0.97*np.dot(np.dot((np.dot(np.transpose(r2),r2))**(-1),np.transpose(r2)),r1) #Estima Rho através de (28) 
        i1w_s=i1w[1:time]-np.dot(ro,i1w[0:time-1])   #Transforma yt*=yt
        i12_s=i12[1:time]-np.dot(ro,i12[0:time-1])   #Transforma xt*=Xt
        B=np.ones((time-1,2))
        B[:,1]=i1w_s
        b_s=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(B),B)),np.transpose(B)),np.transpose(i12_s)) #Regressão de yt* por xt*
        
        b_s[0]=np.divide(b_s[0],1-ro)               # Transforma Beta_0
        rss_s=b_s[0]+np.dot(b_s[1],i1w_s[0:time-1]) # Faz o update da regressão 
        rss_2=b_s[0]+np.dot(b_s[1],i1w[0:time])  
        res_2=i12[0:time]-rss_2
        res_1=res_2[:]                 
    b_ss=b_s

    # Previsão da Corrente I12
    I12f1=beta[0]+np.dot(beta[1],i1w[time:time+timeForecast]) #usando Ordinary Least Squares (OLS))
    I12f2=b_ss[0]+np.dot(b_ss[1],i1w[time:time+timeForecast]) #usando Cochrane-Orcutt (CO))
    print ("Previsão da Corrente I12 considerando OLS:",I12f1,"\n")
    print ("Previsão da Corrente I12 considerando CO:",I12f2)


    x = range(time)
    xx = range(time-1)
    xxx = range(time,time+timeForecast)

    yy1 = i12[0:time]
    yy2 = i1w[0:time]
    yy3 = rss_1
    yy4 = rss_2
    yy5 = i12[0:time]-rss_1
    yy6 = i12[0:time-1]-rss_2[0:time-1]
    yy7 = i12[time:time+timeForecast]
    yy8 = I12f1
    yy9 = I12f2

    plt.plot(xxx, yy7,  color='red', label='Medido')
    plt.plot(xxx, yy8, color='black', linestyle='dashed', marker='o', label='OLS')
    plt.plot(xxx, yy9, color='red', linestyle='-.', marker='*', label='CO')
    plt.legend()
    plt.xlabel('Instante de tempo [h]')
    plt.ylabel('Corrente I_12')
    plt.show()

    plt.plot(yy2, yy1, 'C1o', label='i12')
    plt.plot(yy2, yy3, label='Regressão OLC')
    plt.plot(yy2, yy4, label='Regressão CO')
    plt.legend()
    plt.xlabel("Injeção P_1")
    plt.ylabel("Corrente I_12")
    plt.show()

    plt.plot(x, yy5, 'C1o', label='Resíduos OLS')
    plt.plot(xx, yy6, 'C0*', label='Resíduos CO')
    plt.legend()
    plt.show()

    return I12f1, I12f2

# ============================================================================================================================================
# Funções para Autorregressão AR(1)
# ============================================================================================================================================
def AR1(rss_1, beta, nBus, I, Y, Yl, time, timeForecast):
    # Para calcular I12: usar P1 + o valor anterior da corrente
    ee1 = [ 0.2878,   0.0145,   0.5846,  -0.0029,  -0.2718,  -0.1411,
        -0.2058,  -0.1793,  -0.9878,  -0.4926,  -0.1480,   0.7222,
        -0.3123,   0.4541,   0.9474,  -0.1584,   0.4692,   1.0173,
        -0.0503,   0.4684,  -0.3604,   0.4678,   0.3047,  -1.5098,
        -0.5515,  -0.5159,   0.3657,   0.7160,   0.1407,   0.5424,
            0.0409,   0.0450,   0.2365,  -0.3875,   1.4783,  -0.8487]

    ee =  [ 0.2226,  -0.2293,  -0.1757,  -0.1691,  -0.2031,  -0.5900,
        -0.0556,  -0.0845,  -0.1834,   0.2798,   0.1534,   0.0751,
        -0.1089,   0.3545,   0.0228,  -0.2139,   0.4409,   0.6044,
        -0.2187,  -0.1233,   0.0026,   0.4980,   0.3703,   0.0812,
            0.1183,   0.2486,  -0.0686,  -0.0727,  -0.0009,  -0.1180,
            0.2443,   0.6224,  -0.4600,  -0.3878,   0.4734,  -0.4050]

    II=np.zeros((nBus-1,time+timeForecast), dtype=complex)
    II[:,0]=I
    i12=np.zeros(time+timeForecast)
    i1w=np.zeros(time+timeForecast)

    v=1+np.dot(np.linalg.inv(Yl),I)
    i12[0]=np.absolute(np.dot(Y[0,1],v[0]-v[1]))
    i1w[0]=np.real(I[0])
    for t in range(time+timeForecast-1):
        II[:,t+1]=0.95*II[:,t]+ee[t]
        i1w[t+1]=0.75*i1w[t]+ee1[t]
        II[0,t+1]=i1w[t+1]+np.complex128(0,np.imag(II[0,t+1]))
        v=1+np.dot(np.linalg.inv(Yl),II[:,t+1])
        I12=np.dot(-Y[0,1],v[0]-v[1])
        i12[t+1]=np.absolute(I12)*np.sign(np.real(I12)) 

    # Definir a regressão AR(1)
    # Definir os resíduos ---------------------------- DUVIDAS A PARTIR DAQUI ----------------------------
    res_1=i12[0:time]-rss_1
    for k in range(3):
        r2=res_1[0:time-1]
        r1=res_1[1:time]
        ro=0.97*np.dot(np.dot((np.dot(np.transpose(r2),r2))**(-1),np.transpose(r2)),r1)
        i1w_s=i1w[1:time]-np.dot(ro,i1w[0:time-1])
        i12_s=i12[1:time]-np.dot(ro,i12[0:time-1])
        B=np.ones((time-1,2))
        B[:,1]=i1w_s
        b_s=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(B),B)),np.transpose(B)),np.transpose(i12_s))
        b_s[0]=np.divide(b_s[0],1-ro)
        rss_s=b_s[0]+np.dot(b_s[1],i1w_s[0:time-1])
        rss_2=b_s[0]+np.dot(b_s[1],i1w[0:time])
        res_2=i12[0:time]-rss_2
        res_1=res_2[:]
    
    b_ss=b_s
    
    # Previsão da Corrente I12
    I12f1=beta[0]+np.dot(beta[1],i1w[time:time+timeForecast]) #usando Ordinary Least Squares (OLS))
    I12f2=b_ss[0]+np.dot(b_ss[1],i1w[time:time+timeForecast]) #usando Cochrane-Orcutt (CO))
    print ("Previsão da Corrente I12 considerando OLS:",I12f1,"\n")
    print ("Previsão da Corrente I12 considerando CO:",I12f2)

    # Gráficos
    x = range(time)
    xx = range(time-1)
    xxx = range(time,time+timeForecast)

    yy1 = i12[0:time]
    yy2 = i1w[0:time]
    yy3 = rss_1
    yy4 = rss_2
    yy5 = i12[0:time]-rss_1
    yy6 = i12[0:time-1]-rss_2[0:time-1]
    yy7 = i12[time:time+timeForecast]
    yy8 = I12f1
    yy9 = I12f2

    plt.plot(xxx, yy7,  color='red', label='Medido')
    plt.plot(xxx, yy8, color='black', linestyle='dashed', marker='o', label='OLS')
    plt.plot(xxx, yy9, color='red', linestyle='-.', marker='*', label='CO')
    plt.legend()
    plt.xlabel('Instante de tempo [h]')
    plt.ylabel('Corrente I_12')
    plt.show()

    plt.plot(yy2, yy1, 'C1o', label='i12')
    plt.plot(yy2, yy3, label='Regressão OLC')
    plt.plot(yy2, yy4, label='Regressão CO')
    plt.legend()
    plt.xlabel("Injeção P_1")
    plt.ylabel("Corrente I_12")
    plt.show()

    plt.plot(x, yy5, 'C1o', label='Resíduos OLS')
    plt.plot(xx, yy6, 'C0*', label='Resíduos CO')
    plt.legend()
    plt.show()

    return I12f1, I12

# ============================================================================================================================================
# Funções para Autorregressão AR(1) + Soma de Cargas
# ============================================================================================================================================
def AR1_Soma(i12, i1w, time, timeForecast):
    # NOTA: Para calcular a corrente I12: usar P1 + a soma das cargas + o valor anterior da corrente
    # Definir a regressão AR(1)
    # Definir a regressão AR(1) + Soma de Cargas
    # Definir os resíduos

    return I12f1, I12

# ============================================================================================================================================
# Função Principal
# ============================================================================================================================================
def show_menu():
    """Exibe o menu principal e retorna a opção selecionada pelo usuário."""
    os.system('cls' if os.name == 'nt' else 'clear')  # Limpa a tela
    print("=" * 80)
    print("                 LABORATÓRIO 4 - Regressão sobre Previsão Autoregressiva                 ")
    print("=" * 80)
    print("\nEscolha uma opção:")
    print("0 - Sair")
    print("1 - OLS")
    print("2 - Cochrane-Orcutt (CO)")
    print("3 - AR(1)")
    print("4 - AR(1) + Soma de Cargas")

    try:
        option = int(input("\nOpção: "))
        return option
    except ValueError:
        return -1

def main():
    # Carregar e processar os dados
    SlackBus, Net_Info, Power_Info, Power_Test = load_data('DASG_Prob2_new.xlsx')

    # Preparar a rede
    Y, Yl, G, B, I, nBus = prepare_network(Net_Info, Power_Info, SlackBus)

    # Definir a matriz de erros
    II, i12, i1w = error_matrix(Y, Yl, I, nBus, time, timeForecast)

    while True:
        option = show_menu()
        if option == 0:
            break

        elif option == 1:
            # OLS
            beta, rss_1 = OLS(i12, i1w, time, timeForecast)

        elif option == 2:
            # Cochrane-Orcutt (CO)
            I12f1, I12f2 = CO(rss_1, beta, 0, i12, i1w, time, timeForecast)

        elif option == 3:
            # AR(1)
            I12f1, I12 = AR1(rss_1, beta, 4, I, Y, Yl, time, timeForecast)

        elif option == 4:
            # AR(1) + Soma de Cargas
            I12f1, I12 = AR1_Soma(i12, i1w, time, timeForecast)
        
        else:
            print("Opção inválida. Tente novamente.")

if "__main__":
    main()

