#include "Header.h"

/*********************************************Cliente UDP*****************************************************/
char* cliente_UDP(struct Topologia *dados, char *Mensagem) {
    socklen_t UDP_addrlen;
    struct addrinfo UDP_hints, *UDP_res;
    struct sockaddr_in UDP_addr;
    int errcode;
    char* Mensagem_Recebida = (char*)malloc(512 * sizeof(char));

    if ((dados->Socket_Cliente_UDP = socket(AF_INET, SOCK_DGRAM, 0)) == -1) {
        perror("Falha na criacao do Socket");
        exit(EXIT_FAILURE);
    }

    memset(&UDP_hints, 0, sizeof UDP_hints);
    UDP_hints.ai_family = AF_INET;    /* IPv4 */
    UDP_hints.ai_socktype = SOCK_DGRAM; /* Socket UDP */

    errcode = getaddrinfo(dados->regIp, dados->regUDP, &UDP_hints, &UDP_res);
    if (errcode != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(errcode));
        exit(EXIT_FAILURE);
    }

    if (sendto(dados->Socket_Cliente_UDP, Mensagem, strlen(Mensagem), 0, UDP_res->ai_addr, UDP_res->ai_addrlen) == -1) {
        perror("Falha em enviar Mensagem");
        close(dados->Socket_Cliente_UDP); // Close the UDP socket before exiting
        freeaddrinfo(UDP_res);
        exit(EXIT_FAILURE);
    }

    UDP_addrlen = sizeof(UDP_addr);
    if (recvfrom(dados->Socket_Cliente_UDP, Mensagem_Recebida, 512, 0, (struct sockaddr*)&UDP_addr, &UDP_addrlen) == -1) {
        perror("Falha em receber Mensagem");
        close(dados->Socket_Cliente_UDP); // Close the UDP socket before exiting
        freeaddrinfo(UDP_res);
        free(Mensagem_Recebida); // Free allocated memory before exiting
        exit(EXIT_FAILURE);
    }

    close(dados->Socket_Cliente_UDP); // Close the UDP socket
    freeaddrinfo(UDP_res);

    return Mensagem_Recebida;
}

/*********************************************Servidor TCP****************************************************/
void servidor_TCP(struct Topologia *dados) {
    struct addrinfo TCP_server_hints, *TCP_server_res;

    // Criacao do socket do servidor
    if ((dados->Socket_Servidor_TCP = socket(AF_INET, SOCK_STREAM, 0)) == -1) exit(1);

    // Configuracao das dicas de endereco para o servidor
    memset(&TCP_server_hints, 0, sizeof TCP_server_hints);
    TCP_server_hints.ai_family = AF_INET;
    TCP_server_hints.ai_socktype = SOCK_STREAM;
    TCP_server_hints.ai_flags = AI_PASSIVE;

    if (getaddrinfo(NULL,dados->No.TCP,&TCP_server_hints,&TCP_server_res)!=0) exit(1);

    // Associacao do socket do servidor ao endereco local
    if (bind(dados->Socket_Servidor_TCP,TCP_server_res->ai_addr,TCP_server_res->ai_addrlen) == -1) {
        perror("Falha na execução da função Bind do Servidor");
        close(dados->Socket_Servidor_TCP); // Fechar o socket em caso de erro
        exit(EXIT_FAILURE);
    }

    // Configuracao do socket do servidor para ouvir conexoes
    if (listen(dados->Socket_Servidor_TCP, MAX_CONNECTIONS) == -1) {
        perror("Falha na execução da função Listen do Servidor");
        close(dados->Socket_Servidor_TCP); // Fechar o socket em caso de erro
        exit(EXIT_FAILURE);
    }

    freeaddrinfo(TCP_server_res); 

}

/*********************************************Cliente TCP*****************************************************/
char* cliente_TCP(struct Topologia *dados, char *Mensagem) {
    struct addrinfo TCP_cliente_hints, *TCP_cliente_res;
    char* Mensagem_Recebida = (char*)malloc(512 * sizeof(char));

    dados->Socket_Cliente_TCP = socket(AF_INET, SOCK_STREAM, 0);
    if (dados->Socket_Cliente_TCP == -1) {  
        perror("Falha na criacao do Socket");
        exit(EXIT_FAILURE);
    }

    memset(&TCP_cliente_hints, 0, sizeof TCP_cliente_hints);
    TCP_cliente_hints.ai_family = AF_INET; // IPv4
    TCP_cliente_hints.ai_socktype = SOCK_STREAM; // TCP Socket

    if (getaddrinfo(dados->Suc.ip, dados->Suc.TCP, &TCP_cliente_hints, &TCP_cliente_res) == -1) {
        perror("Falha ao obter informações do endereço");
        exit(EXIT_FAILURE);
    }

    if (connect(dados->Socket_Cliente_TCP, TCP_cliente_res->ai_addr, TCP_cliente_res->ai_addrlen) == -1) {
        perror("Falha na conexão");
        exit(EXIT_FAILURE);
    }

    ssize_t bytes_escritos = write(dados->Socket_Cliente_TCP, Mensagem, strlen(Mensagem));
    if (bytes_escritos <= 0) {
        perror("Falha ao escrever no socket");
        exit(EXIT_FAILURE);
    }

    freeaddrinfo(TCP_cliente_res); 

    return Mensagem_Recebida;
}

/******************************************Cliente Corda TCP**************************************************/
void cliente_corda_TCP(struct Topologia *dados) {
    struct addrinfo TCP_cliente_hints, *TCP_cliente_res;
    char mensagem[128];

    dados->Socket_Cliente_TCP_Corda_enviada = socket(AF_INET, SOCK_STREAM, 0);
    if (dados->Socket_Cliente_TCP_Corda_enviada == -1) {  
        perror("Falha na criacao do Socket");
        exit(EXIT_FAILURE);
    }

    memset(&TCP_cliente_hints, 0, sizeof TCP_cliente_hints);
    TCP_cliente_hints.ai_family = AF_INET; // IPv4
    TCP_cliente_hints.ai_socktype = SOCK_STREAM; // TCP Socket

    if (getaddrinfo(dados->Chord.ip, dados->Chord.TCP, &TCP_cliente_hints, &TCP_cliente_res) == -1) {
        perror("Falha ao obter informações do endereço");
        exit(EXIT_FAILURE);
    }

    if (connect(dados->Socket_Cliente_TCP_Corda_enviada, TCP_cliente_res->ai_addr, TCP_cliente_res->ai_addrlen) == -1) {
        perror("Falha na conexão");
        exit(EXIT_FAILURE);
    }

    sprintf(mensagem, "CHORD %s\n", dados->No.id);

    ssize_t bytes_escritos = write(dados->Socket_Cliente_TCP, mensagem, strlen(mensagem));
    if (bytes_escritos <= 0) {
        perror("Falha ao escrever no socket");
        exit(EXIT_FAILURE);
    }

    freeaddrinfo(TCP_cliente_res); 

}