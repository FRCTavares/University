#include "Header.h"

/****************************************************JOIN*****************************************************/
void join(struct Topologia *dados){

    char Mensagem[512], confirmacao[512];

    sprintf(Mensagem, "ENTRY %s %s %s\n", dados->No.id, dados->No.ip, dados->No.TCP);
    // Verifica se a mensagem contém apenas "NODESLIST anel"
    if (strstr(dados->Nodes_List, "NODESLIST ") != NULL) {
        // Verifica se não há mais nada na mensagem além da linha inicial
        if(sscanf(dados->Nodes_List, "NODESLIST %s", dados->anel) == 1 && strlen(dados->Nodes_List)<15){
            printf("Primeiro no do anel!\n");

            //Predecessor
            strcpy(dados->Pred.id, dados->No.id);
            strcpy(dados->Pred.ip, dados->No.ip);
            strcpy(dados->Pred.TCP, dados->No.TCP);
            //Proprio sucessor
            strcpy(dados->Suc.id, dados->No.id);
            strcpy(dados->Suc.ip, dados->No.ip);
            strcpy(dados->Suc.TCP, dados->No.TCP);
            //Proprio segundo sucessor
            strcpy(dados->Ssuc.id, dados->No.id);
            strcpy(dados->Ssuc.ip, dados->No.ip);
            strcpy(dados->Ssuc.TCP, dados->No.TCP);

            sprintf(Mensagem,"REG %s %s %s %s", dados->anel, dados->No.id, dados->No.ip, dados->No.TCP);
            strcpy(confirmacao, cliente_UDP(dados, Mensagem));
            confirmacao[strlen(confirmacao)] = '\0';
            write(1,confirmacao,5);
            write(1, "\n", 1);

            servidor_TCP(dados);

        }else{    
            /*Escolha de um dos nos da NODESLIST*/
            printf("Escolha um no do anel %s (ID IP TCP) para se ligar: ", dados->anel);
            scanf("%2s %s %s", dados->Suc.id,dados->Suc.ip,dados->Suc.TCP);

            sprintf(Mensagem, "ENTRY %s %s %s\n", dados->No.id, dados->No.ip, dados->No.TCP);
            strcpy(confirmacao, cliente_TCP(dados, Mensagem));
            confirmacao[strlen(confirmacao)] = '\0';
            write(1,confirmacao,strlen(confirmacao));

            servidor_TCP(dados);

        }
    }
}

/*************************************************DIRECT JOIN*************************************************/
void directJoin(struct Topologia *dados){

    char Mensagem[512];

    printf("Qual o anel em que quer introduzir o novo no?\n");
    scanf("%s", dados->anel);

    sprintf(Mensagem, "ENTRY %s %s %s\n", dados->No.id, dados->No.ip, dados->No.TCP);

    if (strcmp(dados->No.id,dados->Suc.id) == 0){
        printf("Primeiro no do anel!\n");

        //Predecessor
        strcpy(dados->Pred.id, dados->No.id);
        strcpy(dados->Pred.ip, dados->No.ip);
        strcpy(dados->Pred.TCP, dados->No.TCP);
        //Proprio sucessor
        strcpy(dados->Suc.id, dados->No.id);
        strcpy(dados->Suc.ip, dados->No.ip);
        strcpy(dados->Suc.TCP, dados->No.TCP);
        //Proprio segundo sucessor
        strcpy(dados->Ssuc.id, dados->No.id);
        strcpy(dados->Ssuc.ip, dados->No.ip);
        strcpy(dados->Ssuc.TCP, dados->No.TCP);

        servidor_TCP(dados);

    }else{
        strcpy(Mensagem, cliente_UDP(dados, Mensagem));
        Mensagem[strlen(Mensagem)] = '\0';
        write(1,Mensagem,5);
        write(1, "\n", 1);

        sprintf(Mensagem, "ENTRY %s %s %s\n", dados->No.id, dados->No.ip, dados->No.TCP);
        strcpy(Mensagem, cliente_TCP(dados, Mensagem));
        Mensagem[strlen(Mensagem)] = '\0';
        write(1,Mensagem,strlen(Mensagem));
        
        servidor_TCP(dados);
    }

}

/****************************************************LEAVE****************************************************/
void leave(struct Topologia *dados){
    
    char Mensagem[256], confirmacao[256];

    sprintf(Mensagem, "UNREG %s %s",dados->anel, dados->No.id); /*Retirar do Registo de Nos o No*/

    strcpy(confirmacao, cliente_UDP(dados, Mensagem));

    strcpy(dados->No.id, "");

    strcpy(dados->Suc.id, "");
    strcpy(dados->Suc.ip, "");
    strcpy(dados->Suc.TCP, "");

    strcpy(dados->Ssuc.id, "");
    strcpy(dados->Ssuc.ip, "");
    strcpy(dados->Ssuc.TCP, "");

    strcpy(dados->Pred.id, "");

}

/************************************************SHOW TOPOLOGY************************************************/
void showtopology(struct Topologia *dados) {
    printf("Informacao do No: \n Identificador: %s\n Endereco IP: %s\n Porta TCP: %s\n\n", dados->No.id, dados->No.ip, dados->No.TCP);
    printf("Informacao do Predecessor do No: \n Identificador: %s\n\n", dados->Pred.id);
    printf("Informacao do Sucessor do No: \n Identificador: %s\n Endereco IP: %s\n Porta TCP: %s\n\n", dados->Suc.id, dados->Suc.ip, dados->Suc.TCP);
    printf("Informacao do Segundo Sucessor do No: \n Identificador: %s\n Endereco IP: %s\n Porta TCP: %s\n\n", dados->Ssuc.id, dados->Ssuc.ip, dados->Ssuc.TCP);
    printf("Corda ligada a: %s\n", dados->Chord.id);
    //Cordas conectadas a este nó...
}

/*************************************************SHOW ROUTING************************************************/
void showRouting(struct Topologia *dados, char *dest){
    for(int i=1; i<16; i++){
        if(strcmp(dados->tabela_encaminhamento[i][0], dest)==0){
            for(int j=1; j<16; j++){
                if(strlen(dados->tabela_encaminhamento[0][j]) > 0){
                    printf("Vizinho: %s \n", dados->tabela_encaminhamento[0][j]);
                    printf("Caminho: %s \n", dados->tabela_encaminhamento[i][j]);
                }
            }
        }
    }
}

/**************************************************SHOW PATH**************************************************/
void showPath(struct Topologia *dados, char *dest){
    for(int i=1; i<16; i++){
        if(strcmp(dados->tabela_caminhos_mais_curtos[i][0], dest)==0){
            printf("Caminho mais curto: %s\n", dados->tabela_caminhos_mais_curtos[i][1]);
        }
    }
}

/***********************************************SHOW FORWARDING***********************************************/
void showForwarding(struct Topologia *dados, char *dest){
    for(int i=1; i<16; i++){
        if(strcmp(dados->tabela_expedicao[i][0], dest)==0){
            printf("Vizinho para obter o caminho mais curto: %s\n", dados->tabela_expedicao[i][1]);
        }
    }
}

/****************************************************CHORD****************************************************/
void chord(struct Topologia *dados){
    char Mensagem[512];

    sprintf(Mensagem, "NODES %s",dados->anel); 
    strcpy(dados->Nodes_List, cliente_UDP(dados, Mensagem));
    dados->Nodes_List[strlen(dados->Nodes_List)] = '\0';

    printf("Escolha um no para ligar uma corda: [ID] [IP] [PORT]");
    sscanf("%s %s %s", dados->Chord.id, dados->Chord.ip, dados->Chord.TCP);

    if((strcmp(dados->Chord.id, dados->Suc.id) != 0) && (strcmp(dados->Chord.id, dados->Pred.id) != 0)){
        cliente_corda_TCP(dados);
    }else{
        printf("Impossivel estabelecer uma corda com o Sucessor/Predecessor. Escolha outro no.");
    }
}

void cria_tabela(struct Topologia *dados){
    //Atualiza tabela de encaminhamento
    char mensagem[512], caminho[256];;

    sprintf(mensagem, "NODES %s",dados->anel); //Retirar do Registo de Nos o No
    strcpy(dados->Nodes_List, cliente_UDP(dados, mensagem));
    dados->Nodes_List[strlen(dados->Nodes_List)] = '\0';

    int col = 0, lin = 0;
    //Guardar a Lista numa tabela
    for (int i = 14; dados->Nodes_List[i] != '\0'; i++){
        if (dados->Nodes_List[i] == '\n'){
            lin++;
            col = 0;
        }else{
            dados->Nodes_List_tabela[lin][col] = dados->Nodes_List[i];
            col++;
        }
    }

    //Preencher a coluna 0 com todo os nós do anel (Destinos)
    int i = 1;
    for (lin = 0; lin < 16; lin++){
        char temp[3];
        sprintf(temp, "%c%c", dados->Nodes_List_tabela[lin][0], dados->Nodes_List_tabela[lin][1]);
        strcpy(dados->tabela_encaminhamento[i][0], temp);
        strcpy(dados->tabela_caminhos_mais_curtos[i][0], temp);
        strcpy(dados->tabela_expedicao[i][0], temp);
        i++;
    }

    //Preenchimento dos vizinhos do nó (Linha 0)
    //Caso em que é o segundo nó
    if ((strcmp(dados->Suc.id, dados->Pred.id) == 0) && (strcmp(dados->Ssuc.id, dados->No.id) == 0)){
        strcpy(dados->tabela_encaminhamento[0][1], dados->Pred.id);

        for(int i=2; i<16; i++){
            strcpy(dados->tabela_encaminhamento[0][i], "");
        }
    //Caso geral
    } else{
        strcpy(dados->tabela_encaminhamento[0][1], dados->Pred.id);
        strcpy(dados->tabela_encaminhamento[0][2], dados->Suc.id);

        for(int i=3; i<16; i++){
            strcpy(dados->tabela_encaminhamento[0][i], "");
        }
    }

    for (int i = 1; i < 16; i++){
        for (int j = 1; j < 16; j++){
            //Caso em que o destino é a origem.
            //Caso em que o destino é o predecessor e o vizinho o sucessor.
            //Caso em que o destino é o sucessor e o vizinho o predecessor.
            if ((strcmp(dados->tabela_encaminhamento[i][0], dados->No.id) == 0) || ((strcmp(dados->tabela_encaminhamento[0][j], dados->Suc.id) == 0) && (strcmp(dados->tabela_encaminhamento[i][0], dados->Pred.id) == 0) && strcmp(dados->Suc.id, dados->Pred.id) != 0) || ((strcmp(dados->tabela_encaminhamento[0][j], dados->Pred.id) == 0) && (strcmp(dados->tabela_encaminhamento[i][0], dados->Suc.id) == 0) && strcmp(dados->Suc.id, dados->Pred.id) != 0)) {
                strcpy(dados->tabela_encaminhamento[i][j], "");

            }else if ((strcmp(dados->tabela_encaminhamento[i][0], dados->tabela_encaminhamento[0][j]) == 0) && (strlen(dados->tabela_encaminhamento[i][0]) != 0) && (strlen(dados->tabela_encaminhamento[0][j]) != 0)){
                sprintf(caminho, "%s-%s", dados->No.id, dados->tabela_encaminhamento[i][0]); 
                strcpy(dados->tabela_encaminhamento[i][j], caminho);
                
                //Manda Route
                sprintf(mensagem, "ROUTE %s %s %s\n", dados->No.id, dados->tabela_encaminhamento[i][0], caminho);

                if (strcmp(dados->tabela_encaminhamento[0][i], dados->Suc.id) == 0){
                    write(dados->Socket_Cliente_TCP, mensagem, strlen(mensagem));
                }else if (strcmp(dados->tabela_encaminhamento[0][i], dados->Pred.id) == 0){
                    write(dados->Socket_Cliente_TCP_Pred, mensagem, strlen(mensagem));
                }
            //Caso Geral em que manda a mensagem route 
            }else if ((strlen(dados->tabela_encaminhamento[i][0]) != 0) && (strlen(dados->tabela_encaminhamento[0][j]) != 0) && (strcmp(dados->tabela_encaminhamento[i][0], dados->No.id) != 0)){
                sprintf(caminho, "%s-%s", dados->No.id, dados->tabela_encaminhamento[0][j]); 
                sprintf(mensagem, "ROUTE %s %s %s\n", dados->No.id, dados->tabela_encaminhamento[i][0], caminho);

                if (strcmp(dados->tabela_encaminhamento[0][i], dados->Suc.id) == 0){
                    write(dados->Socket_Cliente_TCP, mensagem, strlen(mensagem));
                }else if (strcmp(dados->tabela_encaminhamento[0][i], dados->Pred.id) == 0){
                    write(dados->Socket_Cliente_TCP_Pred, mensagem, strlen(mensagem));
                }
            //Caso para as celulas da tabela que ficam por preencher.
            }else{
                strcpy(dados->tabela_encaminhamento[i][j], "");
            }
        }
    }
}

void atualiza_caminhos_mais_curtos_e_expedicao(struct Topologia *dados){
    //Atualiza tabela de caminhos mais curtos
    for(int i=1; i<16; i++){
        for(int j=1; j<16; j++){
            if ((strlen(dados->tabela_encaminhamento[i][j+1]) != 0) && (strlen(dados->tabela_encaminhamento[i][j]) != 0)){
                if (strlen(dados->tabela_encaminhamento[i][j]) > strlen(dados->tabela_encaminhamento[i][j+1])){
                    strcpy(dados->tabela_caminhos_mais_curtos[i][1], dados->tabela_encaminhamento[i][j+1]);
                    strcpy(dados->tabela_expedicao[i][1], dados->tabela_encaminhamento[0][j+1]);
                }else{
                    strcpy(dados->tabela_caminhos_mais_curtos[i][1], dados->tabela_encaminhamento[i][j]);
                    strcpy(dados->tabela_expedicao[i][1], dados->tabela_encaminhamento[0][j]);
                }
            }
        }
    }    

   //Atualiza tabela de expedicao
    for(int i=1; i<16; i++){
        char temp[16];
        strcpy(temp, dados->tabela_caminhos_mais_curtos[i][1]);
        char exp[3]; // Array para armazenar o no vizinho
        exp[0] = temp[3]; 
        exp[1] = temp[4]; 
        exp[2] = '\0'; 
        strcpy(dados->tabela_expedicao[i][1], exp);
    }
}

void imprimir_tabela(struct Topologia *dados){
    //Imprime cada vizinho e os destinos possiveis com os seus caminhos
    printf("Tabela de encaminhamento do No %s\n", dados->No.id);

    for (int j = 0; j < 16; j++){
        if (strlen(dados->tabela_encaminhamento[0][j]) != 0){
            printf("\nVizinho %s: ", dados->tabela_encaminhamento[0][j]);
            for (int i = 0; i < 16; i++){
                if (strlen(dados->tabela_encaminhamento[i][0]) != 0){
                    printf(" %s -> %s |", dados->tabela_encaminhamento[i][0], dados->tabela_encaminhamento[i][j]);
                }
            }
        }
    }
}