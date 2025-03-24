###############################################################################################
# Laboratório 6 - Identificação de fase                                                       #
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
cosPhi=0.95
time=48
m=12
netFactor=0.25
noiseFactor=0.00

# ============================================================================================================================================
# Dados de consumo
# ============================================================================================================================================
def data():
    #Consumption dataset
    s=  [[0.0450,    0.0150,    0.0470,    0.0330],
        [0.0250,    0.0150,    0.2480,    0.0330],
        [0.0970,    0.0250,    0.3940,    0.0330],
        [0.0700,    0.0490,    0.0200,    0.4850],
        [0.1250,    0.0460,    0.0160,    0.1430],
        [0.2900,    0.0270,    0.0160,    0.0470],
        [0.2590,    0.0150,    0.0170,    0.0200],
        [0.2590,    0.0160,    0.0280,    0.0160],
        [0.4420,    0.0160,    0.0500,    0.0170],
        [0.2010,    0.0230,    0.0460,    0.0160],
        [0.2060,    0.0490,    0.0220,    0.0240],
        [0.1300,    0.0470,    0.0160,    0.0490],
        [0.0460,    0.0260,    0.0170,    0.0480]]
    s = np.array(s)

    #topology
    topo=[[1, 2],[2,3],[3,4]]
    nBUS=np.max(topo)

    #Impedance
    z=np.multiply([complex(0.1,0.05),complex(0.15,0.07),complex(0.2,0.1)],netFactor)

    vr=1 #Reference voltage
    el=1
    ni=20 #Iterations for the Power Flow

    return s, topo, nBUS, z, vr, el, ni

# ============================================================================================================================================
# Função para calculo do Power Flow
# ============================================================================================================================================
def pf3ph(t,z,si,vr,el,ni,al):
    #Matrices creation
    t=np.array(t)
    p=t[:,0]
    f=t[:,1]
    w=len(p)+1
    vp=np.zeros((nBUS-1,w), dtype=complex)
    vn=np.zeros((nBUS-1,w), dtype=complex)
    vp[0,0:w]=vr
    
    for h in range (2,nBUS):
        vp[h-1,:]=vp[h-2,:]*al  #Create a three phase system of voltages
                                #Voltages will be the same in all BUS

    va=vp-vn                                                      #Auxiliar voltage
    ia=np.conj(np.divide(np.multiply(si,np.abs(va)**el),va))      #Auxiliar current 
    
    for it in range(ni):                                          #Iterations of Power Flow
        va=vp-vn
        ip=np.conj(np.divide(np.multiply(si,np.abs(va)**el),va))  #Phase current 
        inn=-np.sum(ip,0)                                         #Neutral current 
        for k in range(w-1,0,-1):                                 #Backward Cycle
            n=f[k-1]
            m=p[k-1]
            ip[:,m-1]=ip[:,m-1]+ip[:,n-1]                         #Phase Current
            inn=-np.sum(ip,0)                                     #Neutral Current

        eps= np.linalg.norm(np.max(np.abs(ia-ip),0))              #Error, comparing the new currents and the old ones (previous iteration)

        if eps>1e-4:
            ia=ip
            mvp=0
            mvn=0
            eps=np.inf
        else:                       #If the error is lower than the limit, we can return the results 
            mvp=(vp-vn)             #Phase Voltages to return
            mvn=vn[0,:]             #Neutral Voltage to return
            #return mvp, mvn, eps, ip, inn;
            return mvp;
        for k in range (w-1):                     #Forward Cycle
            n=f[k]                                
            m=p[k]
            vn[:,n-1]=vn[:,m-1]-z[k]*inn[n-1]     #Neutral Voltage 
            vp[:,n-1]=vp[:,m-1]-z[k]*ip[:,n-1]    #Phase Voltage
        ia=ip             #Save the current of previous iteration
    
    return mvp

# ============================================================================================================================================
# Criação das matrizes
# ============================================================================================================================================
def create_matrices():
    #Creation of Matrices
    al=np.exp(np.multiply(np.multiply(complex(0,-1),2/3),np.pi)) #Phase Angle
    Y=np.zeros((3*m), dtype=complex)
    X=np.zeros((3*m,m), dtype=complex)
    v=np.zeros((m,3))
    dv_abs=np.zeros((m,3))


    for i in range(m):
        si=[[0, 0, s[i,2], 0],[0, 0, s[i,1], 0],[0, s[i,0],  0, s[i,3]]] #Connection of consumers by
                                                                        #node and by phase
                                                                        #Consumer 1 (s[i,0]) is 
                                                                        #connected to Bus 2 in Phase 3
        mvp=pf3ph(topo,z,si,vr,el,ni,al)
        noise=1+noiseFactor*np.random.randn(3)
        mvp[:,3]=np.multiply(mvp[:,3],noise)                       #Add noise to the voltages
        Y[3*(i):3*(i)+3]=mvp[:,3]                                  #Save the voltages in matrix Y
        dv_abs[i,:]=vr-np.abs(mvp[:,3])                            #Volage variations (only to plot)

    Volt=np.reshape(Y,(m,3))   

    print ('The voltages measured in the PMUs are:\n',Volt)

    return Y, X, v, dv_abs

# ============================================================================================================================================
# Função para cálculo de fase
# ============================================================================================================================================
def phase_id():
    # Get data and matrices
    s, topo, nBUS, z, vr, el, ni = data()
    Y, X, v, dv_abs = create_matrices()
    
    # Get phase angle (al should be defined here since it's used in this function)
    al = np.exp(np.multiply(np.multiply(complex(0,-1),2/3),np.pi))
    
    # The value of Z is the multiplication between D and W 
    # Convert to numpy array for proper matrix operations
    Z = np.array([[2, al, al**2],
                  [1, 2*al, al**2],
                  [1, al, 2*al**2]])
    
    # Calculate Z inverse for solving the system of equations
    Z_inv = np.linalg.inv(Z)

    # Create matrices for voltage deviations
    vz = np.zeros(3*m, dtype=complex)
    vz_abs = np.zeros(3*m, dtype=complex)

    # Compute voltage deviations for each time period m   
    for i in range(m):
        # PMU info: complex voltage measurements
        vz[3*i:3*i+3] = np.multiply(vr, [1, al, al**2]) - Y[3*i:3*i+3]
        
        # RTU info: magnitude-only measurements
        vz_abs[3*i:3*i+3] = np.multiply(vr-np.abs(Y[3*i:3*i+3]), [1, al, al**2])

    # Initialize beta matrices
    Bpmu = np.zeros((m, 3), dtype=complex)
    Brtu = np.zeros((m, 3), dtype=complex)

    # Compute betas for each measurement time
    for i in range(m):
        # Extract voltage deviations for time i
        vz_i = vz[3*i:3*i+3]
        vz_abs_i = vz_abs[3*i:3*i+3]
        
        # Solve the system Z * beta = v for both PMU and RTU data
        Bpmu[i,:] = np.matmul(Z_inv, vz_i)
        Brtu[i,:] = np.matmul(Z_inv, vz_abs_i)

    # Calculate magnitudes for easier interpretation
    BBpmu = np.abs(Bpmu)
    BBrtu = np.abs(Brtu)

    # Print results
    print('Betas (complex) considering information from PMUs\n', Bpmu, '\n')
    print('Betas (complex) considering information from RTUs\n', Brtu, '\n')

    print('Betas considering information from PMUs\n', BBpmu, '\n')
    print('Betas considering information from RTUs\n', BBrtu, '\n')

    return Bpmu, Brtu, BBpmu, BBrtu


# ============================================================================================================================================
# Função Principal
# ============================================================================================================================================
def show_menu():
    # Pretende exebir o menu principal?
    print("Preciona ENTER para exibir o menu principal.")
    input()

    """Exibe o menu principal e retorna a opção selecionada pelo usuário."""
    os.system('cls' if os.name == 'nt' else 'clear')  # Limpa a tela
    print("=" * 80)
    print("                 LABORATÓRIO 6 - ID de Fase                 ")
    print("=" * 80)
    print("\nEscolha uma opção (essencial que seja por ordem):")
    print("0 - Sair")
    print("1 - ")
    print("2 - ")
    print("3 - ")
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


    rss_1 = 0
    beta = np.zeros(2)

    while True:
        option, show_plots = show_menu()
        if option == 0:
            break

        elif option == 1:    

        elif option == 2:

        elif option == 3:

        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()