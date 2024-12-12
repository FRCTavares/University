#ifndef COMANDOS_H
#define COMANDOS_H

#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <time.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <sys/select.h>
#include <stdbool.h>

/* Porto e IP de registo de contacto do servidor de nós */
#define PORT_TEJO "59000"
#define IP_TEJO "193.136.138.142"
#define MAX_CONNECTIONS 10
#define TEJO_PRIV "192.168.1.1"


// Definições de estruturas de dados
struct Node {
    char id[4];
    char ip[30]; // Endereço IP
    char TCP[20]; // Porta TCP
};
struct Topologia {
    struct Node Pred;
    struct Node No;
    struct Node Suc;
    struct Node Ssuc;
    struct Node Chord;
    int Socket_Cliente_TCP;// Descritor de arquivo do cliente TCP do No
    int Socket_Servidor_TCP; // Descritor de arquivo do servidor TCP do No
    int Socket_Cliente_UDP; 
    int Socket_Cliente_TCP_Pred;
    
    int Socket_Cliente_TCP_Corda_enviada;
    int Socket_Cliente_TCP_Corda_recebida;

    char anel[20];
    char regIp[20];
    char regUDP[20];
    int no_registado;

    char tabela_encaminhamento[16][16][20]; //Tabela de encaminhamento do nó
    char tabela_caminhos_mais_curtos[16][2][20]; //Tabela dos caminhos mais curtos do nó
    char tabela_expedicao[16][2][20]; //Tabela de expedicao do nó

    char Nodes_List[2048]; //String que guarda a lista que recebe do Servidor de Nós
    char Nodes_List_tabela[40][20]; //Tabela que guarda a lista do nos
};



void join(struct Topologia *dados);
void directJoin(struct Topologia *dados);
void leave(struct Topologia *dados);
void showtopology(struct Topologia *dados);

char* cliente_UDP(struct Topologia *dados, char *Mensagem);
char* cliente_TCP(struct Topologia *dados, char *Mensagem);
void servidor_TCP(struct Topologia *dados);
void cliente_corda_TCP(struct Topologia *dados);

void showRouting(struct Topologia *dados, char *dest);
void showPath(struct Topologia *dados, char *dest);
void showForwarding(struct Topologia *dados, char *dest);
void cria_tabela(struct Topologia *dados);
void imprimir_tabela(struct Topologia *dados);
void atualiza_caminhos_mais_curtos_e_expedicao(struct Topologia *dados);

//void imprimir_tabela(struct Topologia *dados);

#endif /* COMANDOS_H */
