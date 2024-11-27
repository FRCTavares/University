#include "Header.h"

/*********************************************************************************************************/
/*COR.c, programa principal a partir de onde se efetuam todas funções da aplicação.                       */
/*                                                                                                       */
/*                                                                                                       */
/* Trabalho desenvolvido por: Marta Valente (103574) e Francisco Tavares (103402)                        */
/*********************************************************************************************************/

int main(int argc, char *argv[]) {

    struct Topologia dados;
/***************************** Invocação da aplicação e inicialização ************************************/
    if (argc > 5 || argc < 3) {
        printf("Número errado de argumentos!");
        return 1; 
    }else{
        dados.Pred = (struct Node){"0", "0", "0"};
        dados.No = (struct Node){"0", "0", "0"};
        dados.Suc = (struct Node){"0", "0", "0"};
        dados.Ssuc = (struct Node){"0", "0", "0"};
        dados.Socket_Cliente_TCP = 0;
        dados.Socket_Servidor_TCP = 0;
        dados.Socket_Cliente_UDP = 0;
        dados.Socket_Cliente_TCP_Pred = 0;
        dados.no_registado = 0;

        // Inicializar tabela_encaminhamento com caracteres vazios
        memset(dados.tabela_encaminhamento, '\0', sizeof(dados.tabela_encaminhamento));
        
        // Inicializar tabela_caminhos_mais_curtos com caracteres vazios
        memset(dados.tabela_caminhos_mais_curtos, '\0', sizeof(dados.tabela_caminhos_mais_curtos));
        
        // Inicializar tabela_expedicao com caracteres vazios
        memset(dados.tabela_expedicao, '\0', sizeof(dados.tabela_expedicao));

        //Inicializar Nodes_List com caracteres vazios
        memset(dados.Nodes_List, '\0', sizeof(dados.Nodes_List));
        
        // Inicializar Nodes_List_tabela com caracteres vazios
        memset(dados.Nodes_List_tabela, '\0', sizeof(dados.Nodes_List_tabela));    

        strcpy(dados.No.ip, argv[1]);
        strcpy(dados.No.TCP, argv[2]);

        if (argc == 3) {
            strcpy(dados.regIp, IP_TEJO);
            strcpy(dados.regUDP, PORT_TEJO);
        } else {
            strcpy(dados.regIp, argv[3]);
            strcpy(dados.regUDP, argv[4]);
        }
    }

/********************************************Loop principal da aplicação*********************************/

    //Este loop ficará ativo enquanto o nó desta aplicação estiver no anel.
    //Fará uso da função Select() para multiplexagem de descritores.
    //Estes descritores serão os seguintes: 
    //Comandos no Teclado, 
    //Novas ligações Clientes->Servidores TCP
    //Mensagens do Sucessor Servidor->Cliente TCP
    //Mensagens do Predecessor Cliente->Servidor TCP
    //Ligações de Cordas ao Servidor TCP
    //Criação de Corda a partir do Cliente TCP

    char Comando[128];  // Assumindo um tamanho máximo de entrada de 256 caracteres
    /*Inicialização das Variáveis Locais*/
    int descritor = 0;
    ssize_t bytes;
    fd_set fds;
    struct sockaddr addr;
    socklen_t addrlen;
    char mensagem[4096];

    while (1){
        
        printf("\nIntroduza um dos seguintes comandos:\n-join (j) ring id\n-direct join (dj) id succid succIP succTCP\n-chord (c)\n-remove chord (rc))");
        printf("\n-show topology (st)\n-show routing (sr) dest\n-show path (sp) dest\n-show forwarding (sf)\n-message (m) dest message\n-leave (l)\n-exit (x)\n\n");
        FD_ZERO(&fds);                                  //Remove todos os descritores.
        FD_SET(0, &fds);                                //Adiciona o descritor da escrita de comandos.
        FD_SET(dados.Socket_Servidor_TCP, &fds);        //Adiciona o descritor do servidor para novas conexões. fd_tcp_server
        FD_SET(dados.Socket_Cliente_TCP, &fds);         //Adiciona o descritor do cliente para receber mensagens do Sucessor. fd_succ
        FD_SET(dados.Socket_Cliente_TCP_Pred, &fds);    //Adiciona o descritor do servidor para recerber mensagens do Predecessor. fd_prev
        FD_SET(dados.Socket_Cliente_TCP_Corda_enviada, &fds);
        FD_SET(dados.Socket_Cliente_TCP_Corda_recebida, &fds);

        descritor = select(FD_SETSIZE, &fds, NULL, NULL, NULL);

        switch (descritor){
            default:

                if ((FD_ISSET(dados.Socket_Servidor_TCP, &fds) != 0) && (dados.Socket_Servidor_TCP > 0)){

                    int Socket_temp = 0;
                    Socket_temp = dados.Socket_Cliente_TCP_Pred;

                    addrlen = sizeof(addr);
                    if ((dados.Socket_Cliente_TCP_Pred = accept(dados.Socket_Servidor_TCP, &addr, &addrlen)) == -1){
                        perror("Falha ao aceitar novo nó!");
                        exit(EXIT_FAILURE);
                    }

                    bytes = read(dados.Socket_Cliente_TCP_Pred, mensagem, sizeof(mensagem));
                    if (bytes == -1){
                        perror("Falha a ler a Mensagem do novo nó!");
                        exit(EXIT_FAILURE);
                    }
                    mensagem[bytes] = '\0';

                    printf("--[Select]: Mensagem do novo nó que se juntou a mim: %s\n", mensagem);

                    //Tratamento das Mensagens Recebidas do novo no
                    char pred_antigo[4];
                    strcpy(pred_antigo, dados.Pred.id);
                    if (sscanf(mensagem, "ENTRY %s %s %s\n", dados.Pred.id, dados.Pred.ip, dados.Pred.TCP) == 3) {
                        if (strcmp(dados.No.id, dados.Suc.id) == 0){
                            //Como neste caso apenas existia lá um nó vai enviar ao novo nó que o seu segundo sucessor é ele mesmo.
                            sprintf(mensagem, "SUCC %s %s %s\n", dados.Pred.id, dados.Pred.ip, dados.Pred.TCP);

                            strcpy(dados.Suc.id, dados.Pred.id);
                            strcpy(dados.Suc.ip, dados.Pred.ip);
                            strcpy(dados.Suc.TCP, dados.Pred.TCP);
           
                            //Envia ao novo no as infromacoes do seu Sucessor, ou seja, informações do novo nó.
                            bytes = write(dados.Socket_Cliente_TCP_Pred, mensagem, strlen(mensagem));
                            if (bytes <= 0) {
                                perror("Falha ao escrever no socket");
                                exit(EXIT_FAILURE);
                            }

                            sleep(1);
                            //Como neste caso o antigo predecessor do nó é ele mesmo então ele próprio vai ligar-se pelo seu cliente ao servidor do novo no e enviar PRED
                            sprintf(mensagem, "PRED %s\n", dados.No.id);
                            strcpy(mensagem, cliente_TCP(&dados, mensagem));
                            printf("Mensagem Enviada para o seu antigo Predecessor!\n");

                            cria_tabela(&dados);

                        }else{
                            if((strcmp(dados.Suc.id, dados.Pred.id) != 0) && (strcmp(dados.No.id, dados.Ssuc.id) != 0)){
                                for(int j=1; j <16; j++){
                                    if(strcmp(dados.tabela_encaminhamento[0][j], pred_antigo) == 0){
                                        strcpy(dados.tabela_encaminhamento[0][j], dados.Pred.id);
                                        for(int i=1; i<16; i++){
                                            strcpy(dados.tabela_encaminhamento[i][j], "");
                                        }
                                    }
                                }
                            }
                            //Neste caso o Segundo sucessor do novo nó equivale ao sucessor do nó a que se junta.
                            sprintf(mensagem, "SUCC %s %s %s\n", dados.Suc.id, dados.Suc.ip, dados.Suc.TCP);
                    
                            bytes = write(dados.Socket_Cliente_TCP_Pred, mensagem, strlen(mensagem));
                            if (bytes <= 0) {
                                perror("Falha ao escrever no socket");
                                exit(EXIT_FAILURE);
                            }

                            // Enviar mensagem ao Predecessor antigo para avisar da entrada do novo no
                            sprintf(mensagem, "ENTRY %s %s %s\n", dados.Pred.id, dados.Pred.ip, dados.Pred.TCP);

                            printf("Enviou aviso ao antigo predecessor a dizer: %s\n", mensagem);

                            bytes = write(Socket_temp, mensagem, strlen(mensagem));
                            if (bytes <= 0) {
                                perror("Falha ao escrever no socket");
                                exit(EXIT_FAILURE);
                            } 
                            Socket_temp = 0;
                        }
                    }else if (sscanf(mensagem, "PRED %s\n", dados.Pred.id) == 1){
                        //Um nó ja existente do anel está se a juntar ao no novo para fechar o anel;
                        //Registo e confirmacao do registo do no no servidor de nos.
                        //system("clear");
                        char confirmacao[1024];

                        if (dados.no_registado == 0){
                            sprintf(mensagem,"REG %s %s %s %s", dados.anel, dados.No.id, dados.No.ip, dados.No.TCP);
                            strcpy(confirmacao, cliente_UDP(&dados, mensagem));
                            confirmacao[strlen(confirmacao)] = '\0';
                            write(1,confirmacao,5);
                            write(1, "\n", 1);

                        }

                        sprintf(mensagem, "SUCC %s %s %s\n", dados.Suc.id, dados.Suc.ip, dados.Suc.TCP);
    
                        //Envia ao novo no as infromacoes do seu Sucessor, ou seja, informações do novo nó.
                        bytes = write(dados.Socket_Cliente_TCP_Pred, mensagem, strlen(mensagem));
                        if (bytes <= 0) {
                            perror("Falha ao escrever no socket");
                            exit(EXIT_FAILURE);
                        }

                        /*************************Encaminhamento*************************/
                        
                        cria_tabela(&dados);
                    
                    //Recebe CHORD
                    }else if(sscanf(mensagem, "CHORD %s\n", dados.Chord.id) == 1){
                        dados.Socket_Cliente_TCP_Corda_recebida = dados.Socket_Cliente_TCP_Pred;
                        printf("Corda recebida do no %s\n", dados.Chord.id);
                    }  
                }

                if ((FD_ISSET(dados.Socket_Cliente_TCP, &fds) != 0) && (dados.Socket_Cliente_TCP > 0)){
                    //Analisa o caso de uma mensagem recebida, que foi enviada pelo meu sucessor

                    bytes = read(dados.Socket_Cliente_TCP, mensagem, sizeof(mensagem));
                    if (bytes == -1){
                        perror("Falha a ler a Mensagem do Sucessor!");
                        exit(EXIT_FAILURE);
                    }
                    mensagem[bytes] = '\0';

                    printf("--[Select]: Mensagem do meu Sucessor: %s\n", mensagem);

                    char aux1[20], aux2[20], aux3[20];
                    char ori[20], dest[20], caminho[256];
                    char rece_id[20], Mensagem_Chat[128];


                    if (sscanf(mensagem, "ENTRY %s %s %s\n", aux1, aux2, aux3) == 3) {
                        if((strcmp(dados.Suc.id, dados.Pred.id) != 0) && (strcmp(dados.No.id, dados.Ssuc.id) != 0)){
                            for(int j=1; j <16; j++){
                                if(strcmp(dados.tabela_encaminhamento[0][j], dados.Suc.id) == 0){
                                    strcpy(dados.tabela_encaminhamento[0][j], aux1);
                                    for(int i=1; i<16; i++){
                                        strcpy(dados.tabela_encaminhamento[i][j], "");
                                    }
                                }
                            }
                        }

                        //Quando o cliente recebe ENTRY tem que fechar a atual ligacao TCP e inciar uma nova com as informacoes que recebeu na mensagem
                        //De seguinda tem que enviar PRED %s\n para esse novo cliente

                        strcpy(dados.Ssuc.id, dados.Suc.id);
                        strcpy(dados.Ssuc.ip, dados.Suc.ip);
                        strcpy(dados.Ssuc.TCP, dados.Suc.TCP);

                        strcpy(dados.Suc.id, aux1);
                        strcpy(dados.Suc.ip, aux2);
                        strcpy(dados.Suc.TCP, aux3);

                        close(dados.Socket_Cliente_TCP);

                        sprintf(mensagem, "PRED %s\n", dados.No.id);

                        strcpy(mensagem, cliente_TCP(&dados, mensagem));

                        // Enviar mensagem ao Predecessor para que ele atualize o segundo sucessor
                        sprintf(mensagem, "SUCC %s %s %s\n", dados.Suc.id, dados.Suc.ip, dados.Suc.TCP);

                        bytes = write(dados.Socket_Cliente_TCP_Pred, mensagem, strlen(mensagem));
                        if (bytes <= 0) {
                            perror("Falha ao escrever no socket");
                            exit(EXIT_FAILURE);
                        } 
                        
                    }else if (sscanf(mensagem, "SUCC %s %s %s\n", dados.Ssuc.id, dados.Ssuc.ip, dados.Ssuc.TCP) == 3){
                        //printf("Atualização de Segundo sucessor: id - %s, ip - %s, tcp - %s\n", dados.Ssuc.id, dados.Ssuc.ip, dados.Ssuc.TCP);
                        //Nó recebeu mensagem para atualizar o seu segundo sucessor.
                    }else if (strlen(mensagem) == 0){
                        // O meu sucessor saiu do anel

                        //atualiza tabela de encaminhamento
                        for(int i=1; i<16; i++){
                            for(int j=1; j<16; j++){
                                if(strcmp(dados.tabela_encaminhamento[i][0], dados.Suc.id)==0){
                                    for(int k=i; k<(16-i); k++){
                                        strcpy(dados.tabela_encaminhamento[i][0], dados.tabela_encaminhamento[i+1][0]);
                                    }
                                }else if(strcmp(dados.tabela_encaminhamento[0][j], dados.Suc.id)==0){
                                    strcpy(dados.tabela_encaminhamento[0][j], "");
                                }
                            }
                        }

                        close(dados.Socket_Cliente_TCP);
                        FD_CLR(dados.Socket_Cliente_TCP, &fds);
                        dados.Socket_Cliente_TCP = 0;

                        if (strcmp(dados.Suc.id, dados.Pred.id) == 0){ // O anel só tinha dois nós

                            strcpy(dados.Suc.id, dados.Ssuc.id);
                            strcpy(dados.Suc.ip, dados.Ssuc.ip);
                            strcpy(dados.Suc.TCP, dados.Ssuc.TCP);   

                            strcpy(dados.Pred.id, dados.Ssuc.id);
                            strcpy(dados.Pred.ip, dados.Ssuc.ip);
                            strcpy(dados.Pred.TCP, dados.Ssuc.TCP); 

                        } else if (strcmp(dados.Pred.id, dados.Ssuc.id) == 0){ // O anel só tinha três nós

                            strcpy(dados.Suc.id, dados.Ssuc.id);
                            strcpy(dados.Suc.ip, dados.Ssuc.ip);
                            strcpy(dados.Suc.TCP, dados.Ssuc.TCP);   

                            strcpy(dados.Ssuc.id, dados.No.id);
                            strcpy(dados.Ssuc.ip, dados.No.ip);
                            strcpy(dados.Ssuc.TCP, dados.No.TCP);   

                            //Avisa o seu predecessor que tem um novo segundo sucessor -> Enviar para o Predecessor SUCC
                            sprintf(mensagem, "SUCC %s %s %s\n", dados.Suc.id, dados.Suc.ip, dados.Suc.TCP);
                    
                            bytes = write(dados.Socket_Cliente_TCP_Pred, mensagem, strlen(mensagem));
                            if (bytes <= 0) {
                                perror("Falha ao escrever no socket");
                                exit(EXIT_FAILURE);
                            }

                            //Inicia uma nova ligacao com o seu antigo segundo sucessor 
                            sprintf(mensagem, "PRED %s\n", dados.No.id);
                            strcpy(mensagem, cliente_TCP(&dados, mensagem));

                        }else{ // O anel tem quatro ou mais nós
                            //Atualiza o sucessor para as informacoes do antigo Segundo sucessor
                            strcpy(dados.Suc.id, dados.Ssuc.id);
                            strcpy(dados.Suc.ip, dados.Ssuc.ip);
                            strcpy(dados.Suc.TCP, dados.Ssuc.TCP);  

                            //Avisa o seu predecessor que tem um novo segundo sucessor -> Enviar para o Predecessor SUCC
                            sprintf(mensagem, "SUCC %s %s %s\n", dados.Suc.id, dados.Suc.ip, dados.Suc.TCP);
                    
                            bytes = write(dados.Socket_Cliente_TCP_Pred, mensagem, strlen(mensagem));
                            if (bytes <= 0) {
                                perror("Falha ao escrever no socket");
                                exit(EXIT_FAILURE);
                            }
                            sleep(1);

                            //Inicia uma nova ligacao com o seu antigo segundo sucessor 
                            sprintf(mensagem, "PRED %s\n", dados.No.id);
                            strcpy(mensagem, cliente_TCP(&dados, mensagem));

                        }   
                    }else if (sscanf(mensagem, "ROUTE %s %s %s\n", ori, dest, caminho) == 3){
                        if((strcmp(dest, dados.No.id)==0)){
                            sprintf(mensagem, "ROUTE %s %s %s\n", ori, dest, caminho);
                            write(dados.Socket_Cliente_TCP, mensagem, strlen(mensagem));

                        //No de origem recebe ROUTE de volta com vetor de encaminhamento completo
                        }else if(strcmp(ori, dados.No.id)==0){
                            for (int i = 1; i < 16; i++){
                                for (int j = 1; j < 16; j++){
                                    if((strcmp(dados.tabela_encaminhamento[i][0], dest)==0) && (strcmp(dados.tabela_encaminhamento[0][j], dados.Suc.id)==0)){
                                        strcpy(dados.tabela_encaminhamento[i][j], caminho);
                                    }
                                }
                            }
                        }else{
                            char caminho_atualizado[1024];
                            sprintf(caminho_atualizado, "%s-%s", caminho, dados.Pred.id); //atualiza a info do caminho
                            sprintf(mensagem, "ROUTE %s %s %s\n", ori, dest, caminho_atualizado);
                            write(dados.Socket_Cliente_TCP_Pred, mensagem, strlen(mensagem));
                        }
                        
                        if(strcmp(ori, dados.No.id) != 0){
                            int linha, coluna;
                            for(int i=1; i<16; i++){
                                if(strcmp(dados.tabela_encaminhamento[i][0], "") == 0){
                                    strcpy(dados.tabela_encaminhamento[i][0], ori);
                                    linha=i;
                                    break;
                                }
                            }

                            for(int j=1; j<16; j++){
                                if((strcmp(dados.Suc.id, ori)==0) && (strcmp(dados.tabela_encaminhamento[0][j], "")==0)){
                                    strcpy(dados.tabela_encaminhamento[0][j], ori);
                                    coluna=j;
                                    break;
                                }else if(strcmp(dados.Suc.id, dados.tabela_encaminhamento[0][j]) == 0){
                                    coluna=j;
                                    break;                                    
                                }
                            }
                            
                            //Inverte caminho
                            char *token = strtok(caminho, "-");
                            int i=strlen(caminho);
                            char caminho_invertido[i*3-1];
                            caminho_invertido[0] = '\0';
                            while(token != NULL && i>=0){
                                char temp[4];
                                sprintf(temp, "%s-", token);
                                strcat(temp, caminho_invertido);
                                strcpy(caminho_invertido, temp);
                                token = strtok(NULL, "-");
                            }

                            caminho_invertido[strlen(caminho_invertido) - 1] = '\0';

                            strcpy(dados.tabela_encaminhamento[linha][coluna], caminho_invertido);
                        }

                    }else if (sscanf(mensagem, "CHAT %s %s\n", rece_id, Mensagem_Chat) == 2){
                        if (strcmp(rece_id, dados.No.id) == 0){
                            printf("Mensagem recebida: %s\n", Mensagem_Chat);
                        }else{
                            write(dados.Socket_Cliente_TCP_Pred, mensagem, sizeof(mensagem));
                        }
                    }
                }

                if ((FD_ISSET(dados.Socket_Cliente_TCP_Pred, &fds) != 0) && (dados.Socket_Cliente_TCP_Pred > 0)){

                    bytes = read(dados.Socket_Cliente_TCP_Pred, mensagem, sizeof(mensagem));
                    if (bytes == -1){
                        perror("Falha a ler a Mensagem do Predecessor!");
                        exit(EXIT_FAILURE);
                    }
                    mensagem[bytes] = '\0';

                    printf("--[Select]: Mensagem recebida do predecessor: %s\n", mensagem);
                    
                    char ori[20], dest[20], caminho[256];
                    char rece_id[20], Mensagem_Chat[128];

                    if (strlen(mensagem) == 0){
                        //atualiza tabela de encaminhamento
                        for(int i=1; i<16; i++){
                            for(int j=1; j<16; j++){
                                if(strcmp(dados.tabela_encaminhamento[i][0], dados.Pred.id)==0){
                                    for(int k=i; k<(16-i); k++){
                                        strcpy(dados.tabela_encaminhamento[i][0], dados.tabela_encaminhamento[i+1][0]);
                                    }
                                }else if(strcmp(dados.tabela_encaminhamento[0][j], dados.Pred.id)==0){
                                    strcpy(dados.tabela_encaminhamento[0][j], "");
                                }
                            }
                        }

                        printf("A ligação TCP foi desligada\n");
                        close(dados.Socket_Cliente_TCP_Pred);
                        FD_CLR(dados.Socket_Cliente_TCP_Pred, &fds);
                        dados.Socket_Cliente_TCP_Pred = 0;

                    }else if (sscanf(mensagem, "ROUTE %s %s %s\n", ori, dest, caminho) == 3){

                        //Chegou ao destino
                        if((strcmp(dest, dados.No.id)==0)){
                            sprintf(mensagem, "ROUTE %s %s %s\n", ori, dest, caminho);
                            write(dados.Socket_Cliente_TCP_Pred, mensagem, strlen(mensagem));
                        
                        //No de origem recebe ROUTE de volta com vetor de encaminhamento completo
                        }else if(strcmp(ori, dados.No.id)==0){
                            for (int i = 1; i < 16; i++){
                                for (int j = 1; j < 16; j++){
                                    if((strcmp(dados.tabela_encaminhamento[i][0], dest)==0) && (strcmp(dados.tabela_encaminhamento[0][j], dados.Pred.id)==0)){
                                        strcpy(dados.tabela_encaminhamento[i][j], caminho);
                                    }
                                }
                            }

                        //Caso geral
                        }else{
                            char caminho_atualizado[1024];
                            sprintf(caminho_atualizado, "%s-%s", caminho, dados.Suc.id); //atualiza a info do caminho
                            sprintf(mensagem, "ROUTE %s %s %s\n", ori, dest, caminho_atualizado);
                            write(dados.Socket_Cliente_TCP, mensagem, strlen(mensagem));
                        }   

                        if(strcmp(ori, dados.No.id) != 0){

                            int linha, coluna;
                            for(int i=1; i<16; i++){
                                if(strcmp(dados.tabela_encaminhamento[i][0], "") == 0){
                                    strcpy(dados.tabela_encaminhamento[i][0], ori);
                                    linha=i;
                                    break;
                                }
                            }

                            for(int j=1; j<16; j++){
                                if((strcmp(dados.Pred.id, ori)==0) && (strcmp(dados.tabela_encaminhamento[0][j], "")==0)){
                                    strcpy(dados.tabela_encaminhamento[0][j], ori);
                                    coluna=j;
                                    break;
                                }else if(strcmp(dados.Pred.id, dados.tabela_encaminhamento[0][j]) == 0){
                                    coluna=j;
                                    break;                                    
                                }
                            }
                            
                            //Inverte caminho
                            char *token = strtok(caminho, "-");
                            int i=strlen(caminho);
                            char caminho_invertido[i*3-1];
                            caminho_invertido[0] = '\0';
                            while(token != NULL && i>=0){
                                char temp[4];
                                sprintf(temp, "%s-", token);
                                strcat(temp, caminho_invertido);
                                strcpy(caminho_invertido, temp);
                                token = strtok(NULL, "-");
                            }

                            caminho_invertido[strlen(caminho_invertido) - 1] = '\0';

                            strcpy(dados.tabela_encaminhamento[linha][coluna], caminho_invertido);
                        }
                  
                    }else if (sscanf(mensagem, "CHAT %s %s\n", rece_id, Mensagem_Chat) == 2){
                        if (strcmp(rece_id, dados.No.id) == 0){
                            printf("Mensagem recebida: %s\n", Mensagem_Chat);
                        }else{
                            write(dados.Socket_Cliente_TCP, mensagem, sizeof(mensagem));
                        }
                    }
                }     
                
                if ((FD_ISSET(dados.Socket_Cliente_TCP_Corda_enviada, &fds) != 0) && (dados.Socket_Cliente_TCP_Corda_enviada > 0)){

                    bytes = read(dados.Socket_Cliente_TCP_Corda_enviada, mensagem, sizeof(mensagem));
                    if (bytes == -1){
                        perror("Falha a ler a Mensagem da Corda!");
                        exit(EXIT_FAILURE);
                    }
                    mensagem[bytes] = '\0';

                    printf("--[Select]: Mensagem recebida da Corda: %s\n", mensagem);

                    if (strlen(mensagem) == 0){
                        printf("A minha corda partiu!\n");
                        //atualiza tabela de encaminhamento
                        for(int i=1; i<16; i++){
                            for(int j=1; j<16; j++){
                                if(strcmp(dados.tabela_encaminhamento[i][0], dados.Suc.id)==0){
                                    for(int k=i; k<(16-i); k++){
                                        strcpy(dados.tabela_encaminhamento[i][0], dados.tabela_encaminhamento[i+1][0]);
                                    }
                                }else if(strcmp(dados.tabela_encaminhamento[0][j], dados.Suc.id)==0){
                                    strcpy(dados.tabela_encaminhamento[0][j], "");
                                }
                            }
                        }

                        close(dados.Socket_Cliente_TCP_Corda_enviada);
                        FD_CLR(dados.Socket_Cliente_TCP_Corda_enviada, &fds);
                        dados.Socket_Cliente_TCP_Corda_enviada = 0;

                    }
                }

                if ((FD_ISSET(dados.Socket_Cliente_TCP_Corda_recebida, &fds) != 0) && (dados.Socket_Cliente_TCP_Corda_recebida > 0)){

                    bytes = read(dados.Socket_Cliente_TCP_Corda_enviada, mensagem, sizeof(mensagem));
                    if (bytes == -1){
                        perror("Falha a ler a Mensagem da Corda!");
                        exit(EXIT_FAILURE);
                    }
                    mensagem[bytes] = '\0';

                    printf("--[Select]: Mensagem recebida da Corda: %s\n", mensagem);

                    if (strlen(mensagem) == 0){
                        printf("A minha corda partiu!\n");
                        //atualiza tabela de encaminhamento
                        for(int i=1; i<16; i++){
                            for(int j=1; j<16; j++){
                                if(strcmp(dados.tabela_encaminhamento[i][0], dados.Suc.id)==0){
                                    for(int k=i; k<(16-i); k++){
                                        strcpy(dados.tabela_encaminhamento[i][0], dados.tabela_encaminhamento[i+1][0]);
                                    }
                                }else if(strcmp(dados.tabela_encaminhamento[0][j], dados.Suc.id)==0){
                                    strcpy(dados.tabela_encaminhamento[0][j], "");
                                }
                            }
                        }

                        close(dados.Socket_Cliente_TCP_Corda_recebida);
                        FD_CLR(dados.Socket_Cliente_TCP_Corda_recebida, &fds);
                        dados.Socket_Cliente_TCP_Corda_recebida = 0;

                    }
                }

                if (FD_ISSET(0, &fds) != 0){
                    fgets(Comando, sizeof(Comando), stdin);
                    printf("--[Select]: Comando introduzido: %s\n", Comando);
                    Comando[strcspn(Comando, "\n")] = '\0';

                    char dest[4], Mensagem_Chat[128];

                    if ((strncmp(Comando, "j", 1) == 0)  && (dados.Socket_Servidor_TCP == 0)){
                        if (sscanf(Comando, "j %3s %2s", dados.anel, dados.No.id) == 2) {
                            if (strlen(dados.anel) == 3 && strlen(dados.No.id) == 2) { 
                                char Mensagem[512];

                                sprintf(Mensagem, "NODES %s",dados.anel); /*Retirar do Registo de Nos o No*/
                                strcpy(dados.Nodes_List, cliente_UDP(&dados, Mensagem));
                                dados.Nodes_List[strlen(dados.Nodes_List)] = '\0';

                                int col = 0, lin = 0;
                                //Guarda a Lista numa tabela
                                for (int i = 14; dados.Nodes_List[i] != '\0'; i++){
                                    if (dados.Nodes_List[i] == '\n'){
                                        lin++;
                                        col = 0;
                                    }else{
                                        dados.Nodes_List_tabela[lin][col] = dados.Nodes_List[i];
                                        col++;
                                    }
                                }

                                char id_lista[20][3], temp[3];
                                for (lin = 0; lin < 20; lin++){
                                    sprintf(temp, "%c%c", dados.Nodes_List_tabela[lin][0], dados.Nodes_List_tabela[lin][1]);
                                    strncpy(id_lista[lin], temp, 2);
                                    id_lista[lin][2] = '\0';
                                }

                                //system("clear");
                                write(1, dados.Nodes_List, strlen(dados.Nodes_List));
                                write(1, "\n", 1);

                                // Verifica se o novo ID já está na lista se já, escolhe outro id que não esteja na lista e avisa o utilizador
                                bool input_valido = false;
                                while(!input_valido){
                                    input_valido = true;
                                    for(int i=0; i<20; i++){
                                        if(strcmp(dados.No.id, id_lista[i])==0){
                                            printf("O ID %s já está no anel.", id_lista[i]);
                                            input_valido = false;
                                            break;
                                        }
                                    }

                                    if(input_valido){
                                        break;
                                    
                                    //Escolha aleatória de um novo id
                                    }else{
                                        srand(time(NULL));
                                        int random_id = (rand() % 99) + 1;
                                        sprintf(dados.No.id, "%02d", random_id);
                                        printf("Novo ID escolhido: %s\n", dados.No.id);
                                    }
                                }
                
                                // Chamada da função de junção de um novo nó ao anel escolhido
                                join(&dados);

                            } else {
                                system("clear");
                                printf("Formato inválido. Usar 'j [ringId] [nodeID]'.\n");
                            }
                        }
                    } else if ((strncmp(Comando, "dj", 2) == 0)  && (dados.Socket_Servidor_TCP == 0)){
                        if (sscanf(Comando, "dj %2s %2s %15s %5s", dados.No.id, dados.Suc.id, dados.Suc.ip, dados.Suc.TCP) == 4) {
                            if (strlen(dados.No.id) == 2 && strlen(dados.Suc.id) == 2 && strlen(dados.Suc.TCP) == 5) {
                                directJoin(&dados);
                            } else {
                                system("clear");
                                printf("Formato inválido. Usar 'dj [nodeId] [successorId] [successorIP] [successorPort]'.\n");
                            }
                        }
                    } else if ((strncmp(Comando, "c", 1) == 0) && (dados.no_registado == 1)) {
                        printf("A adição de uma corda não está funcional.\n");
                    } else if ((strncmp(Comando, "rc", 2) == 0)  && (dados.no_registado == 1)){
                        printf("A remoção de uma corda não está funcional.\n");
                    } else if (strncmp(Comando, "st", 2) == 0) {
                        showtopology(&dados);
                    } else if (strncmp(Comando, "sr", 2) == 0) {
                        if(sscanf(Comando, "sr %2s", dest)==1){
                            showRouting(&dados, dest);
                        }
                    } else if (strncmp(Comando, "sp", 2) == 0) {
                        if(sscanf(Comando, "sp %s", dest)==1){
                            atualiza_caminhos_mais_curtos_e_expedicao(&dados);
                            showPath(&dados, dest);
                        }
                    } else if (strncmp(Comando, "sf", 2) == 0) { 
                        if(sscanf(Comando, "sf %s", dest)==1){
                            atualiza_caminhos_mais_curtos_e_expedicao(&dados);
                            showForwarding(&dados, dest);
                        }
                    } else if (strncmp(Comando, "m", 1) == 0){
                        printf("entrou aqui\n");
                        if (sscanf(Comando, "m %2s %s", dest, Mensagem_Chat) == 2) {
                            if (strcmp(dados.No.id, dest) != 0){
                                printf("A enviar a mensagem: '%s' do nó%s ao nó%s\n", Mensagem_Chat, dados.No.id, dest);

                                sprintf(mensagem, "CHAT %s %s\n", dest, Mensagem_Chat);
                                //Sem considerar o caminho mais curto
                                write(dados.Socket_Cliente_TCP, mensagem, sizeof(mensagem));
                                //message(&dados);
                            }else{
                                printf("Não pode escolher como destino o proprio no!\n");
                            }
                        }
                    } else if (strncmp(Comando, "l", 1) == 0) {
                        Comando[0] = '\0';
                        if (dados.Socket_Cliente_TCP != 0){
                            close(dados.Socket_Cliente_TCP);
                            FD_CLR(dados.Socket_Cliente_TCP, &fds);
                            dados.Socket_Cliente_TCP = 0;
                        }
                        if (dados.Socket_Servidor_TCP != 0){
                            close(dados.Socket_Servidor_TCP);
                            FD_CLR(dados.Socket_Servidor_TCP, &fds);
                            dados.Socket_Servidor_TCP = 0;
                        }
                        if (dados.Socket_Cliente_TCP_Pred != 0){
                            close(dados.Socket_Cliente_TCP_Pred);
                            FD_CLR(dados.Socket_Cliente_TCP_Pred, &fds);
                            dados.Socket_Cliente_TCP_Pred = 0;
                        }

                        leave(&dados);
                        dados.no_registado = 0;
                    } else if (strncmp(Comando, "x", 1) == 0) {
                        exit(1);
                    } else {
                        printf("Comando inválido ou se escolheu join ou direct join quer dizer que esta aplicação já tem um nó incializado.\n");
                    }
                }
        }
    }
}