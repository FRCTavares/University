#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
/*
|--------------------------------------------------------------------------------------------------------------------------|
|                                                    Projeto Final de Programação                                          |
|                                                                LEEC                                                      |
|                                                             2021/2022                                                    |
|                                                                                                                          |
|  Trabalho realizado por: Marta Valente (103574) e Francisco Tavares (103402)                                             |
|  Grupo: 117                                                                                                              |
|                                                                                                                          |
|  Este trabalho corresponde na implemetação de uma versão particular de um jogo de palavras cruzadas. Este jogo           |
|  implementa vários modos de jogo, tanto para jogadores humanos, como para estratégias de jogo executadas pelo            |
|  computador.                                                                                                             |
|                                                                                                                          |
|                                                                                                                          |
|__________________________________________________________________________________________________________________________|                                                                                                                          |
*/

//Variáveis Globais

char a[16][16]; //matriz para o tabuleiro
int pontos=0;
FILE *dic_desorg; //dicionario desorganizado
char *dic_aloc[150000]; //dicionario alocado
char *dic_organ[150000]; //dicionario organizado


void criartabela2(int n){ // Função para criar o tabuleiro inicial vazio.
/* Criamos esta função com o objetivo de escrever o tabuleiro inicial dependendo do tamanho inserido pelo utilizador. */

    int x=(n/2 + 1),i ,j; // variável auxiliar para colocação do '3' e do '#', i: número da linha, j: número da coluna.

    for (i = 1; i <= n; i++){

        if (i<10){
            printf(" %d ", i); // Coloca os algarismos das linhas antes do tabuleiro e separa nos algarismo inferiores e ,superiores e iguas, a 10 para que o tabuleiro
        }else{                 // fique um quadrado sem linhas maiores do que outras.
            printf("%d ",i);
        }
        for (j = 1; j <= n; j++)
        {
            if ((i==1 && j==1) || (i==n && j==1) || (i==n && j==n) || (i==1 && j==n)){ // Posiciona o simbolo '$'nos cantos do tabuleiro.
                a[i][j]='$';
                printf ("%c ", a[i][j]);
            }
            else if ((i==n && j==n) || (j+i==n+1) || (j==i)){// Posiciona o número '2' no tabuleiro.
                a[i][j]='2';
                printf ("%c ",a[i][j]);
            }
            else if ((i==1 && j==x) || (i==n && j==x) || (i==x && j==1) || (i==x && j==1) || (i==x && j==n)){ // Posiciona o número '3' no tabuleiro.
                a[i][j]='3';
                printf("%c ", a[i][j]);
            }
            else if (((j==2 && (i==x-1 || i==x+1)) || ((i==2 && (j==x-1 || j==x+1))) || ((j==n-1 &&(i==x-1 || i==x+1))) || (i==n-1 &&(j==x-1 || j==x+1)))){// Posciona o simbolo '#' no tabuleiro.
                a[i][j]='#';
                printf("%c ", a[i][j]);
            }
            else{
                a[i][j]='.'; //Posciona o simbolo '.' no tabuleiro.
                printf("%c ", a[i][j]);
            }
        }
            printf("\n");
    }

    char aux;
    printf("   A ");
    for (aux = 'B'; aux < 'B'+n-1; aux++){ // Coloca as letras representantes das colunas por baixo do tabuleiro.
        printf("%c ",aux);
    }
return;
}

void criartab2(int coluna, int linha, int n, int tamanho_palavra, int ori, char palavra[]){// Criar tabela com a palavra escolhida pelo utilizador nas coordenadas também indicadas
/*Nesta função ao contrário da primeira já escrevemos a palavra da jogada no tabuleiro respeitando todas as restrições definidas anteriormente em validacao1_2 e em validacaoHori2
e em validacaoVerti2*/

    int lin, col, letra, i, j;

    if (ori==0){// Escrita da palavra na horizontal.
        letra=0;
        for (col=coluna; col<=coluna+tamanho_palavra-1; col++){
            a[linha][col]=palavra[letra];
            letra++;
        }
    }else if (ori==1){// Escrita da palavra na vertical.
        letra=0;
        for (lin=linha; lin<=linha+tamanho_palavra-1; lin++){
            a[lin][coluna]=palavra[letra];
            letra++;
        }
    }


    for (i = 1; i <= n; i++)
    {
        if (i<10){
            printf(" %d ", i); // Coloca os algarismos das linhas antes do tabuleiro.
        }else{
            printf("%d ",i);
        }
        for (j = 1; j <= n; j++){//Coloca no tabuleiro o resto dos caracteres '.' '3' '2' '$' '#'.
            printf("%c ", a[i][j]);
        }
    printf("\n");
    }

    char aux;
    printf("   A ");
    for (aux = 'B'; aux < 'B'+n-1; aux++){ // Coloca as letras representantes das colunas por baixo do tabuleiro.
        printf("%c ",aux);
    }
    printf("\n\n");
return;
}

void pontuacao2(int ori, int coluna, int linha, int tamanho_palavra, char palavra[]){ // Função para pontuar a jogada.
    int j=0, l=0, i=0, pontosMAIS=0, multiplicacao=0, duplo=0;
    char multi[16];

    /* Na parte inicial da função pontuacao o espaço onde a palavra vai ser colocada no tabuleiro vai ser verificado para serem econtrados possíveis bónus atribuidos depois ao array multi[]
    para mais tarde multiplicar por cada letra ou pela palavra todo no caso do '$' a pontução. Utilizámos um array multi[] para mais facilmente atribuir a cada letra esse bónus podendo esse
    ser 1 caso o simbolo lido seja um '.' ou uma letra, no caso em que ´é acrescentada a uma palavra ja existente letras.
    */

    if (ori==0){//pontuação nas posições do tabuleiro em que se encontram os seguintes caracteres: '.' , '2', '3', '#', '$' (quando a jogada é feita na horizontal)
        for (j=coluna; j<coluna+tamanho_palavra; j++){
            if (a[linha][j]=='.'){
                multi[l]=1;
                l++;
            }else if(a[linha][j]=='2'){
                multi[l]=2;
                l++;
            }else if(a[linha][j]=='3'){
                multi[l]=3;
                l++;
            }else if(a[linha][j]=='$'){
                multi[l]=1;
                duplo++;
                l++;
            }else{
                multi[l]=1;
                l++;
            }
        }
    }else if (ori==1){//pontuação nas posições do tabuleiro em que se encontram os seguintes caracteres: '.' , '2', '3', '#', '$' (quando a jogada é feita na vertical)
        for (j=linha; j<linha+tamanho_palavra; j++){
            if (a[j][coluna]=='2'){
                multi[l]=2;
                l++;
            }else if (a[j][coluna]=='3'){
                multi[l]=3;
                l++;
            }else if (a[j][coluna]=='$'){
                multi[l]=1;
                duplo++;
                l++;
            }else if (a[j][coluna]=='.'){
                multi[l]=1;
                l++;
            }else{
                multi[l]=1;
                l++;
            }
        }
    }

    j=0;
    l=0;
    i=0;
    pontos=0;

    while(i<tamanho_palavra){ // atribuição dos pontos de cada letra.

    /* Resumidamente, a palavra é lida letra a letra e a cada letra é atribuido um valor de pontos definidos no enuciado. Depois esses pontos são multiplicados
    pelo fator de miultiplicação definido anteriormente no array multi[] que guardou um fator para cada letra na posição selecionada.*/

        if ((palavra[i]=='a') || (palavra[i]=='e') || (palavra[i]=='i') || (palavra[i]=='l') || (palavra[i]=='n') || (palavra[i]=='o') || (palavra[i]=='r') || (palavra[i]=='s') || (palavra[i]=='t') || (palavra[i]=='u')){
            pontosMAIS = 1;
            i++;
            multiplicacao=multi[l];
            l++;
        }else if ((palavra[i]=='b') || (palavra[i]=='c') || (palavra[i]=='m') || (palavra[i]=='p')){
            pontosMAIS = 3;
            i++;
            multiplicacao=multi[l];
            l++;
        }else if ((palavra[i]=='d') || (palavra[i]=='g')){
            pontosMAIS = 2;
            i++;
            multiplicacao=multi[l];
            l++;
        }else if ((palavra[i]=='f') || (palavra[i]=='h') || (palavra[i]=='v') || (palavra[i]=='w') || (palavra[i]=='y')){
            pontosMAIS = 4;
            i++;
            multiplicacao=multi[l];
            l++;
        }else if ((palavra[i]=='j') || (palavra[i]=='x')){
            pontosMAIS = 8;
            i++;
            multiplicacao=multi[l];
            l++;
        }else if (palavra[i]=='k'){
            pontosMAIS = 5;
            i++;
            multiplicacao=multi[l];
            l++;
        }else if ((palavra[i]=='q') || (palavra[i]=='z')){
            pontosMAIS = 10;
            i++;
            multiplicacao=multi[l];
            l++;
        }
       pontos = pontos + ( multiplicacao * pontosMAIS);
    }

    if (duplo==1){// mutiplica os pontos totais da palavra caso anteriormente tenham sido verificados um ou mais símbolos '$'.
         pontos = 2 * pontos;
    }
    if(duplo==2){
        pontos = 4 * pontos;
    }
return;
}

int validacao1_2(int ori, int coluna, int linha, int tamanho_palavra, int n){ //Função para validar a colocação a primeira jogada
/* Nesta função verificamos se a primeira palavra inserida é colocada na horizontal e uma das letras dessa palavra coincide com a posição central do tabuleiro*/
    int erro=0;

    if (ori==0){
        if (tamanho_palavra+coluna<=n+1){
            if (((tamanho_palavra+coluna)>=((n+1)/2)) && (linha==((n+1)/2))){// Posição inicial da palavra (TEM QUE PREENCHER O CENTRO DO TABULEIRO).
                erro=0;
            }else{
                erro= 1;
            }
        }else{
            erro= 1;
        }
    }else{
        erro= 1;
    }

return erro;

}

int validacaoHoriz2(int coluna, int linha, int tamanho_palavra, char palavra[], int n){
/*Esta função permite verificar se a palavra a introduzir nas coordenadas escolhidas pelo computador a partir da segunda jogada são válidas. Verifica se a palavra nas coordenadas escolhida
sai fora do tabueiro (caso sim erro=1). Verifica se a palavra nas coordenas escolhidas coincide com um simbolo proíbido '#' (caso sim erro=1). Verifica se a palavra nas coordenadas escolhidas
utiliza pelo menos uma letra anteriormente colocada no tabuleiro (caso sim erro=0). Verifica se a palavra nas coordenadas escolhidas utiliza pelo menos uma letra nova (caso sim erro=0).
Verifica se a palavra nas coordenadas escolhidas fica ao lado de uma letra (No primeira letra e na ultima) que não forma uma palavra válida (caso sim, erro=1). Verifica se alguma letra da palavra nas
coordenas escolhidas substitui uma letra já inserida anteriomente no tabuleiro diferente da a inserir,(caso a letra a substituir seja diferente erro=1)*/

    int es=0, palavrasiguaisteste=0,i=0;
    int vali=0;

    for (int col=coluna; col<coluna+tamanho_palavra; col++){// Verifica se a palavra a introduzir não colide com nenhum simbolo proíbido '#'.
        if (a[linha][col]=='#'){
            vali++;
        }else{
            vali++;
            vali--;
        }
    }
    for (int col=coluna; col<coluna+tamanho_palavra; col++){// Verifica se a palavra a ser introduzida utiliza pelo menos uma das letras já existentes no tabuleiro.
        if (palavra[i] == a[linha][col]){
            palavrasiguaisteste++;
        }
        i++;
    }
    if ((palavrasiguaisteste==tamanho_palavra) || (palavrasiguaisteste==0)){//Verifica se a palavra a ser introduzida não é igual à anterior nas mesmas coordendas, ou seja, se não esta a escrever uma palavra igual por cima de uma ja existente.
        vali++;
    }

    for (int col=coluna; col<tamanho_palavra+coluna; col++){
        if ((col==coluna) && (col!=1)){
            if ((a[linha][col-1]!='.') && (a[linha][col-1]!='2') && (a[linha][col-1]!='3') && (a[linha][col-1]!='$') && (a[linha][col-1]!='#')){ //Verifica se o simbolo anterior às coordenadas inicias da palavra é um '.' , '2', '3', '#' e '$'.
                vali++;
            }
        }
        if ((col==tamanho_palavra+coluna-1) && (col!=n)){
            if ((a[linha][col+1]!='.') && (a[linha][col+1]!='2') && (a[linha][col+1]!='3') && (a[linha][col+1]!='$') && (a[linha][col+1]!='#')){ //Verifica se o simbolo a seguir às coordenadas finais da palavra é um '.' , '2', '3', '#' e '$'.
                vali++;
            }
        }
    }

    if (tamanho_palavra + coluna <= n+1){//Verifica se o tamanho da palavra nas coordenadas escolhidas não ultrapassa o limite do tabuleiro.
        vali++;
        vali--;
        for (es=0; es<tamanho_palavra; es++){
            if ((a[linha][coluna+es]!='.') && (a[linha][coluna+es]!='2') && (a[linha][coluna+es]!='3') && (a[linha][coluna+es]!='$') && (a[linha][coluna+es]!='#')){
                if (palavra[es] == a[linha][coluna+es]){//Verifica se a palavra a introduzir não substitui uma letra já existente naquela posição do tabuleiro.
                    vali++;
                    vali--;
                }else{
                    vali++;
                }
            }
        }
    }else{
        vali++;
    }

    if (vali==0){
        return 0;
    }else{
        return 1;
    }

}

int validacaoVerti2(int coluna, int linha, int tamanho_palavra, char palavra[], int n){
/*Esta função permite verificar se a palavra a introduzir nas coordenadas escolhidas pelo computador a partir da segunda jogada são válidas. Verifica se a palavra nas coordenadas escolhida
sai fora do tabueiro (caso sim erro=1). Verifica se a palavra nas coordenas escolhidas coincide com um simbolo proíbido '#' (caso sim erro=1). Verifica se a palavra nas coordenadas escolhidas
utiliza pelo menos uma letra anteriormente colocada no tabuleiro (caso sim erro=0). Verifica se a palavra nas coordenadas escolhidas utiliza pelo menos uma letra nova (caso sim erro=0).
Verifica se a palavra nas coordenadas escolhidas fica ao lado de uma letra (No primeira letra e na ultima) que não forma uma palavra válida (caso sim, erro=1). Verifica se alguma letra da palavra nas
coordenas escolhidas substitui uma letra já inserida anteriomente no tabuleiro diferente da a inserir,(caso a letra a substituir seja diferente erro=1)*/

    int es=0, palavrasiguaitest=0, i=0;
    int vali=0;

    for (int lin=linha; lin<linha+tamanho_palavra; lin++){// Verifica se a palavra a introduzir não colide com nenhum simbolo proíbido '#'.
        if (a[lin][coluna]=='#'){
            vali++;
        }else{
            vali++;
            vali--;
        }
    }
    for (int lin=linha; lin<linha+tamanho_palavra; lin++){// Verifica se a palavra a ser introduzida utiliza pelo menos uma das letras já existentes no tabuleiro.
        if (palavra[i] == a[lin][coluna]){
            palavrasiguaitest++;
        }
        i++;
    }
    if ((palavrasiguaitest==tamanho_palavra) || (palavrasiguaitest==0)){//Verifica se a palavra a ser introduzida não é igual à anterior nas mesmas coordendas, ou seja, se não esta a escrever uma palavra igual por cima de uma ja existente.
        vali++;
    }

    for (int lin=linha; lin<tamanho_palavra+linha; lin++){
        if ((lin==linha) && (lin!=1)){
            if ((a[lin-1][coluna]!='.') && (a[lin-1][coluna]!='2') && (a[lin-1][coluna]!='3') && (a[lin-1][coluna]!='$') && (a[lin-1][coluna]!='#')){//Verifica se o simbolo anterior às coordenadas inicias da palavra é um '.' , '2', '3', '#' e '$'.
                vali++;
            }
        }
        if ((lin==tamanho_palavra+linha-1) && (lin!=n)){
            if ((a[lin+1][coluna]!='.') && (a[lin+1][coluna]!='2') && (a[lin+1][coluna]!='3') && (a[lin+1][coluna]!='$') && (a[lin+1][coluna]!='#')){//Verifica se o simbolo a seguir às coordenadas finais da palavra é um '.' , '2', '3', '#' e '$'.
                vali++;
            }
        }
    }

    if (tamanho_palavra + linha<=n+1){//Verifica se o tamanho da palavra nas coordenadas escolhidas não ultrapassa o limite do tabuleiro.
        vali++;
        vali--;
        for (es=0; es<tamanho_palavra; es++){
            if ((a[linha+es][coluna]!='.') && (a[linha+es][coluna]!='2') && (a[linha+es][coluna]!='3') && (a[linha+es][coluna]!='$') && (a[linha+es][coluna]!='#')){
                if (palavra[es] == a[linha+es][coluna]){//Verifica se a palavra a introduzir não substitui uma letra já existente naquela posição do tabuleiro.
                    vali++;
                    vali--;
                }else{
                    vali++;
                }
            }
        }
    }else{
        vali++;
    }


    if (vali==0){
        return 0;
    }else{
        return 1;
    }
}

void colocacao_palavras2(int n, char **dic_organ, int pal){
/* Nesta função nós implementamos um algoritmo que testa todas as palavras do dicionário em todas as coordenadas do tabuleiro e tanto na vertical como na horizontal e descartamos
todas as combinações que não são validadas pelas funcões validacao1_2 no caso de ser a primeira jogada, validacaoHori2 no caso de ser uma tentativa de colocar a palavra na horizontal
e validacaoVerti2 no caso de ser uma tentativa de colocar uma palavra na vertical. Caso as funcoes validacaoHori2 ou validacaoVerti2 retornem valor 0 a funcao pontuacao2 será utilizada
para calcular a pontução dessa jogada. Se atribuir mais pontos a essa jogada válida do que à jogada com atribuição máxima de pontos anterior então a jogada atual vai substituir a anterior
como sendo a jogada válida mais pontuada, caso contrário será descartada. Quando o algoritmo terminar de ler todas as palavras do dicionário utilizado então o algoritmo irá immprimir a jogada
guardada como sendo a que atribui o máximo de pontos à jogada e continurá em loop até que rep=!0 já que quando já não encontra palavras que caibam no tabuleiro a jogada com pontução máximo da jogada anterior
não se irá alterar logo rep++ e por isso termina o jogo e mostra a pontução total do jogo.*/

    int i=0;
    int jogada=1;
    int erro=0;
    int rep=0;
    int pontostotais=0;


    //Estas sao as variaveis a testar.
    char *palavra;
    int tamanho_palavra;
    int coluna, linha, ori;

    //Estas sao as variaveis que fornecem a pontuacao maxima.
    char *palavraM;
    int tamanho_palavraM;
    int colunaM, linhaM, oriM;
    char coordCM;
    int coordLM;
    char coordOM;
    int pontosM=0;


    //Inicio do algoritmo.
    for (i=0;i<pal;i++){
        if (i==(pal-1)){
            if (rep!=0){
                criartab2(colunaM, linhaM, n, tamanho_palavraM, oriM, palavraM);
                printf(" Jogada %d: %c%d%c %s, %d pontos\n \n", jogada, coordCM, coordLM, coordOM, palavraM, pontosM);
                erro=0;
                jogada++;
                pontostotais = pontostotais + pontosM;
                palavraM=NULL;
                tamanho_palavraM=0;
                colunaM=0;
                linhaM=0;
                oriM=0;
                pontosM=0;
                i=0;
                rep=0;
            }else{
                printf("Fim do jogo!\n");
                printf("Pontos Totais: %d", pontostotais);
                return;
            }
        }
        palavra=dic_organ[i];
        tamanho_palavra=strlen(palavra);
            for (int k=1; k<=n ;k++){
                 for (int j=1; j<=n ;j++){
                    linha=j;
                    coluna=k;

                    //Para a jogada na Horizontal.
                    ori=0;
                    if (jogada != 1 && ori==0){
                        erro = validacaoHoriz2(coluna, linha, tamanho_palavra, palavra, n);
                    }else{
                        erro = validacao1_2(ori, coluna, linha, tamanho_palavra, n);
                    }

                    if (erro!=1){
                        pontuacao2(ori, coluna, linha, tamanho_palavra, palavra);
                        if (pontos>pontosM){
                            rep++;
                            colunaM=coluna;
                            linhaM=linha;
                            coordCM=64+coluna;
                            coordLM=linha;
                            coordOM='H';
                            pontosM=pontos;
                            palavraM=palavra;
                            oriM=ori;
                            tamanho_palavraM=tamanho_palavra;

                        }
                    }

                    //Para a jogada na Vertical.
                    ori=1;
                    if (jogada!=1 && ori==1){
                        erro = validacaoVerti2(coluna, linha, tamanho_palavra, palavra, n);
                    }else{
                        erro = validacao1_2(ori, coluna, linha, tamanho_palavra, n);
                    }
                    if (erro!=1){
                        pontuacao2(ori, coluna, linha, tamanho_palavra, palavra);
                        if (pontos>pontosM){
                            rep++;
                            colunaM=coluna;
                            linhaM=linha;
                            coordCM=64+coluna;
                            coordLM=linha;
                            coordOM='V';
                            pontosM=pontos;
                            palavraM=palavra;
                            oriM=ori;
                            tamanho_palavraM=tamanho_palavra;

                        }
                    }

                }
            }
        }
    }

void mododejogo2(int n, char *dicionario){

char  c[30], s[30];
    int tamanho;// n numero de linhas/colunas;
    char *ptr;
    //static int  n=9;

    dic_desorg = fopen(dicionario, "r");

    if(NULL == dic_desorg){
        printf(" Erro na leitura do dicionario! \n");
    }

/*
    Nesta operação de 'for' temos o objetivo de retirar do ficheiro de dicionário quaisquer palavras que não
    utilizem as letras de 'a' a 'z' ou que contenham espaços ou que sejam maiores do que o tamanho das linhas e
    das colunas do tabuleiro ou ainda que sejam menores do que 2.
*/
    for (int k=0; k<150000; k++){ // variavel 'k' igual a todos os caracteres de cada palavra do dicionário.

        fgets(c, 30, dic_desorg);
        sscanf(c,"%s ",c);
        //sscanf(c,"%s/0 ",c);
        sscanf(c,"%s\t ",c);
        sscanf(c,"%s\n ",c);
        sscanf(c,"%s/",c);
        ptr = strchr(s,'/');
        if (ptr!=NULL){
            *ptr='\0';
        }
        int tamanho=strlen(c);


        int fail=0;

        for (int w=0; w < tamanho ; w++){
            if (((c[w]<'a') || (c[w]>'z')) || ((tamanho>n) || (tamanho<2))){ //Descarta as palavras do dicionario que nao contem os caracteres pertencentes ao intervalo entre {a,z}.
                fail=-1;                                                     //e descarta palavras com tamanho menor que 2 ou maiores que o tamanho do tabuleiro.
            }
        }

        if (fail==0){                                      //Escreve as palavras no dicionário alocado em que os caracteres pertencem ao intervalo de 'a' a 'z', que sejam maiores
            dic_aloc[k] = (char*) malloc(strlen(c)+1);     //ou iguais a 2 e que sejam menores ou iguais ao tamanho do tabuleiro.
            strcpy(dic_aloc[k],c);
        }
    }

/*
    Aqui apenas utilizamos a sugestão dada no enunciado de organizar as palavras por tamanho começando
    nas maiores e terminando nas mais pequenas.
*/
    int p=0;
    for (int z=n; z>=2; z--){
        for (int k=0; k<150000; k++){

            if (dic_aloc[k]!=NULL){
                strcpy(s, dic_aloc[k]);
                tamanho=strlen(s);

                if (tamanho==z){
                    dic_organ[p]=dic_aloc[k]; //Sempre que o algoritmo encontra uma palavra de tamanho igual escreve-a no novo dicionário organizado de maneira decrescente.
                    p++;
                }
            }
        }
    }

    int pal=0;
    while (dic_organ[pal]!=NULL){ //função que permite saber qual o número total de palavras no dicionário a utilizar no algoritmo 'colocacao_palavras2'.
        pal++;
    }

    criartabela2(n);
    printf("\n\n Regras do jogo:\n (.) casa vazia, onde pode ser posta qualquer letra.\n (#) casa proibida, onde nao se pode por nenhuma letra.\n (2) casa onde a letra la colocada vale o dobro dos pontos.\n (3) casa onde a letra la colocada vale o triplo dos pontos.\n ($) casa onde a palavra la colocada vale o dobro dos pontos.");
    printf("\n\n");
    colocacao_palavras2(n, dic_organ, pal);


    return ;
}

void criartabela1(int n){ // Função para criar o tabuleiro inicial.
    //Criamos esta função com o objetivo de escrever o tabuleiro inicial dependendo do tamanho inserido pelo utilizador.

    int x=(n/2 + 1),i ,j; // variável auxiliar para colocação do '3' e do '#', i: número da linha, j: número da coluna.

    for (i = 1; i <= n; i++)
    {
        if (i<10){
            printf(" %d ", i); // Coloca os algarismos das linhas antes do tabuleiro e separa os algarismos inferiores e, superiores e iguais, a 10 para que o tabuleiro
        }else{                 //fique um quadrado sem linhas maiores do que outras.
            printf("%d ",i);
        }
        for (j = 1; j <= n; j++)
        {
            if ((i==1 && j==1) || (i==n && j==1) || (i==n && j==n) || (i==1 && j==n)){ // Posiciona o simbolo '$' no tabuleiro.
                a[i][j]='$';
                printf ("%c ", a[i][j]);
            }
            else if ((i==n && j==n) || (j+i==n+1) || (j==i)){// Posiciona o número '2' no tabuleiro.
                a[i][j]='2';
                printf ("%c ",a[i][j]);
            }
            else if ((i==1 && j==x) || (i==n && j==x) || (i==x && j==1) || (i==x && j==1) || (i==x && j==n)){ // Posiciona o número '3' no tabuleiro.
                a[i][j]='3';
                printf("%c ", a[i][j]);
            }
            else if (((j==2 && (i==x-1 || i==x+1)) || ((i==2 && (j==x-1 || j==x+1))) || ((j==n-1 &&(i==x-1 || i==x+1))) || (i==n-1 &&(j==x-1 || j==x+1)))){// Posciona o simbolo '#' no tabuleiro.
                a[i][j]='#';
                printf("%c ", a[i][j]);
            }
            else{
                a[i][j]='.'; //Posciona o simbolo '.' no tabuleiro.
                printf("%c ", a[i][j]);
            }

        }
            printf("\n");
    }
    char aux;
    printf("   A ");
    for (aux = 'B'; aux < 'B'+n-1; aux++){ // Coloca as letras representantes das colunas por baixo do tabuleiro.
        printf("%c ",aux);
    }
    return;
}

void criartab1(int coluna, int linha, int n, int tamanho_palavra, int ori, char palavra[], char coord[]){// Criar tabela com a palavra escolhida pelo utilizador nas coordenadas também indicadas
    /* Esta função cria o tabuleiro e escreve a palavra escolhida pelo o utilizador, caso seja aceite, ou seja, caso cumpra os requesitos necessários para ser escrita.
    A palavra será escrita se não colidir com um simbolo proíbido '#'. O resto das restrições foram verifiacadas em 'mododejogo1'*/

    int lin, col, letra, i, j;

    if (ori==0){// Escrita da palavra na horizontal.
        letra=0;
        for (col=coluna; col<coluna+tamanho_palavra; col++){
            if (a[linha][col]=='#'){
                printf("A palavra nao pode estar por cima de um '#'!");
                return;
            }
            a[linha][col]=palavra[letra];
            letra++;
        }
    }else if (ori==1){// Escrita da palavra na vertical.
        letra=0;
        for (lin=linha; lin<linha+tamanho_palavra; lin++){
            if (a[lin][coluna]=='#'){
                printf("A palvra nao pode estar por cima de um '#'!");
                return;
            }
            a[lin][coluna]=palavra[letra];
            letra++;
        }
    }


    for (i = 1; i <= n; i++)
    {
        if (i<10){
            printf(" %d ", i); // Coloca os algarismos das linhas antes do tabuleiro.
        }else{
            printf("%d ",i);
        }
        for (j = 1; j <= n; j++){//Coloca no tabuleiro o resto dos caracteres '.' '3' '2' '$' '#'.
            printf("%c ", a[i][j]);
        }
    printf("\n");
    }

    char aux;
    printf("   A ");
    for (aux = 'B'; aux < 'B'+n-1; aux++){ // Coloca as letras representantes das colunas por baixo do tabuleiro.
        printf("%c ",aux);
    }
    printf("\n Os pontos atribuidos a '%s' nas coordenadas '%s' foram: %d", palavra, coord, pontos);
}

void pontuacao(int ori, int coluna, int tamanho_palavra, int linha, char palavra[]){ // Função para pontuar a jogada.
    int j=0, l=0, i=0, pontosMAIS=0, multiplicacao=0, tamanho_duplo=0, d=0;
    char multi[16], duplo[3];

    /* Na parte inicial da função pontuacao o espaço onde a palavra vai ser colocada no tabuleiro vai ser verificado para serem econtrados possíveis bónus atribuidos depois ao array multi[]
    para mais tarde multiplicar por cada letra ou pela palavra todo no caso do '$' a pontução. Utilizámos um array multi[] para mais facilmente atribuir a cada letra esse bónus podendo esse
    ser 1 caso o simbolo lido seja um '.' ou uma letra, no caso em que ´é acrescentada a uma palavra ja existente letras.
    */

    if (ori==0){//pontuação nas posições do tabuleiro em que se encontram os seguintes caracteres: '.' , '2', '3', '#', '$' (quando a jogada é feita na horizontal)
        for (j=coluna; j<coluna+tamanho_palavra; j++){
            if (a[linha][j]=='.'){
                multi[l]=1;
                l++;
            }else if(a[linha][j]=='2'){
                multi[l]=2;
                l++;
            }else if(a[linha][j]=='3'){
                multi[l]=3;
                l++;
            }else if(a[linha][j]=='$'){
                multi[l]=1;
                duplo[d]=1;
                d++;
                l++;
            }else{
                multi[l]=1;
                l++;
            }
        }
    }else if (ori==1){//pontuação nas posições do tabuleiro em que se encontram os seguintes caracteres: '.' , '2', '3', '#', '$' (quando a jogada é feita na vertical)
        for (j=linha; j<linha+tamanho_palavra; j++){
            if (a[j][coluna]=='2'){
                multi[l]=2;
                l++;
            }else if (a[j][coluna]=='3'){
                multi[l]=3;
                l++;
            }else if (a[j][coluna]=='$'){
                multi[l]=1;
                duplo[d]=1;
                d++;
                l++;
            }else if (a[j][coluna]=='.'){
                multi[l]=1;
                l++;
            }else{
                multi[l]=1;
                l++;
            }
        }
    }

    j=0;
    l=0;
    i=0;

    pontos=0;
    while(i<tamanho_palavra){ // atribuição dos pontos de cada letra.
    /* Resumidamente, a palavra é lida letra a letra e a cada letra é atribuido um valor de pontos definidos no enuciado. Depois esses pontos são multiplicados
    pelo fator de miultiplicação definido anteriormente no array multi[] que guardou um fator para cada letra na posição selecionada.*/



        if ((palavra[i]=='a') || (palavra[i]=='e') || (palavra[i]=='i') || (palavra[i]=='l') || (palavra[i]=='n') || (palavra[i]=='o') || (palavra[i]=='r') || (palavra[i]=='s') || (palavra[i]=='t') || (palavra[i]=='u')){
            pontosMAIS = 1;
            i++;
            multiplicacao=multi[l];
            l++;
        }else if ((palavra[i]=='b') || (palavra[i]=='c') || (palavra[i]=='m') || (palavra[i]=='p')){
            pontosMAIS = 3;
            i++;
            multiplicacao=multi[l];
            l++;
        }else if ((palavra[i]=='d') || (palavra[i]=='g')){
            pontosMAIS = 2;
            i++;
            multiplicacao=multi[l];
            l++;
        }else if ((palavra[i]=='f') || (palavra[i]=='h') || (palavra[i]=='v') || (palavra[i]=='w') || (palavra[i]=='y')){
            pontosMAIS = 4;
            i++;
            multiplicacao=multi[l];
            l++;
        }else if ((palavra[i]=='j') || (palavra[i]=='x')){
            pontosMAIS = 8;
            i++;
            multiplicacao=multi[l];
            l++;
        }else if (palavra[i]=='k'){
            pontosMAIS = 5;
            i++;
            multiplicacao=multi[l];
            l++;
        }else if ((palavra[i]=='q') || (palavra[i]=='z')){
            pontosMAIS = 10;
            i++;
            multiplicacao=multi[l];
            l++;
        }
       pontos = pontos + ( multiplicacao * pontosMAIS);
    }

    tamanho_duplo=strlen(duplo);
    if (tamanho_duplo==2){// mutiplica os pontos totais da palavra caso anteriormente tenham sido verificados um ou mais símbolos '$'.
         pontos = 2 * pontos;
    }
    if(tamanho_duplo==3){
        pontos = 4 * pontos;
    }
    return;
}

void mododejogo1(int linhas){

    static int n;

    // criar um tabuleiro com o valor introduzido na opcao 't'.
    n=linhas;
    criartabela1(n);

    printf("\n\n Regras do jogo:\n (.) casa vazia, onde pode ser posta qualquer letra.\n (#) casa proibida, onde nao se pode por nenhuma letra.\n (2) casa onde a letra la colocada vale o dobro dos pontos.\n (3) casa onde a letra la colocada vale o triplo dos pontos.\n ($) casa onde a palavra la colocada vale o dobro dos pontos.");


// Escrever a coordenada e a palavra.

    char coord[5]; // coordenadas incias da escrita da palavra.
    char palavra[16];// palavra a colocar  no tabuleiro.
    char colunachar;
    int  coluna=1, linha, ori, tamanho_coord, tamanho_palavra, jogada=1, fim=1;//coordenadas inciais reduzidas a numeros do tabuleiro, tamanho da palavra.

    while (fim!=0){ //Escrita de várias palavras.
        int erro=1;

        printf("\n\n Escreva as coordenadas inicias seguidas de um espaco e da palavra que pretende introduzir: ");
        scanf("%s %s", coord, palavra);

        tamanho_coord=strlen(coord);
        tamanho_palavra=strlen(palavra);

            if (coord[0]<'A' || coord[0]>'Z'){

                coord[0]=coord[0]-32;
            }



            for (colunachar='A'; colunachar!=coord[0]; colunachar++){
                coluna++;
            }

            if (tamanho_coord==3){ // Caso em que as coordenadas da linha sao menores do que 10
                linha = coord[1]-'0';// A linha incial de escrita da palavra
                if (coord[2]=='h' || coord[2]=='v'){
                    ori = coord[2]-32;
                    coord[2]=coord[2]-32;
                }else{
                    ori = coord[2];
                }
            }else if(tamanho_coord==4){ // Caso em que as coordenadas da linha sao maiores ou iguais a 10
                linha = 10 + (coord[2]-'0');// Coluna incial da escrita da palavra
                if (coord[3]=='h' || coord[3]=='h'){
                    ori = coord[3]-32;
                    coord[3]=coord[3]-32;
                }else{
                    ori = coord[3];
                }

            }else{
                printf("\n Erro nas coordenadas!");
                return;
            }



            if ((ori=='H') || (ori=='h')){// Definição da orientação da palavra vertical ou horizontal.
                ori=0;
            }else if ((ori=='V') || (ori=='v')){
                ori=1;
            }else{
                printf("\n As coordenadas introduzidas nao sao validas!");// Se o utilizador escrever outro caracter sem ser 'H' ou 'V' dá erro.
            }



            if(jogada==1){// Posição inicial da palavra (TEM QUE PREENCHER O CENTRO DO TABULEIRO).
                if (ori==0){
                    if ((((tamanho_palavra+coluna)<((n+1)/2)) || (coluna>((n+1)/2))) || (linha!=((n+1)/2))){
                            printf("\n A palavra incial tem de preencher o centro do tabuleiro, tente outra vez.");

                            continue;
                            jogada=0;
                        }
                }else
                    if (ori==1){
                        if ((((tamanho_palavra+linha)<((n+1)/2)) || (linha>((n+1)/2))) || (coluna!=((n+1)/2))){
                            printf("\n A palavra incial tem de preencher o centro do tabuleiro, tente outra vez.");

                            continue;
                            jogada=0;
                        }
                    }

            }

    /* Esta função verifica se a palavra nas coordenadas escolhidas colide com uma letra já existente no tabuleiro diferente da letra dessa palavra nessa posição, para além disso esta
    função verifica se a palavra a introduzir utiliza pelo menos uma das letras já existentes no tabuleiro.*/

    if (jogada != 1){
        if (ori==0){
            for (int es=0; es<=tamanho_palavra ;es++){
                if (palavra[es] == a[linha][coluna+es]){
                    erro=0;
                }
            }
            for (int es=0; es<=tamanho_palavra ;es++){
                    if ((a[linha][coluna+es]!='.') && (a[linha][coluna+es]!='2') && (a[linha][coluna+es]!='3') && (a[linha][coluna+es]!='$') && (a[linha][coluna+es]!='#')){
                        if (palavra[es] != a[linha][coluna+es]){
                            erro=1;
                        }
                    }
            }
        }

        if (ori==1){
            for (int es=0; es<=tamanho_palavra ;es++){
                if (palavra[es] == a[linha+es][coluna]){
                    erro=0;
                }
            }
            for (int es=0; es<=tamanho_palavra ;es++){
                if ((a[linha+es][coluna]!='.') && (a[linha+es][coluna]!='2') && (a[linha+es][coluna]!='3') && (a[linha+es][coluna]!='$') && (a[linha+es][coluna]!='#')){
                    if (palavra[es] != a[linha+es][coluna]){
                        erro=1;
                    }
                }
            }
        }


    if (erro==1){
        printf("\n Pelo menos uma das letras da palavra que escolheu nessas coordenadas tem que coincidir com outra na mesma coordenada!");
        jogada=jogada-1;
        return;
    }

        if (erro==0){
            if (ori==0){// Validação do tamanho da palavra consoante o espaço do tabuleiro.
                if ((tamanho_palavra + coluna)<=n+1){
                    pontuacao(ori, coluna, tamanho_palavra, linha, palavra);
                    criartab1(coluna, linha, n, tamanho_palavra, ori, palavra, coord);//Escrever a nova tabela com a palavra
                    coluna=1;
                    linha=0;
                }else{
                    printf("\n As coordenadas que escolheu nao sao validas, pois a palavra sai fora do tabuleiro!");
                }
            }else if (ori==1){
                if ((tamanho_palavra + linha)<=n+1){
                    pontuacao(ori, coluna, tamanho_palavra, linha, palavra);
                    criartab1(coluna, linha, n, tamanho_palavra, ori, palavra, coord);//Escrever a nova tabela com a palavra
                    coluna=1;
                    linha=0;
                }else{
                    printf("\n As coordenadas que escolheu nao sao validas, pois a palavra sai fora do tabuleiro!");
                }
            }
        }
    }else{
        if (ori==0){// Validação do tamanho da palavra consoante o espaço do tabuleiro.
            if ((tamanho_palavra + coluna)<=n+1){
                pontuacao(ori, coluna, tamanho_palavra, linha, palavra);
                criartab1(coluna, linha, n, tamanho_palavra, ori, palavra, coord);//Escrever a nova tabela com a palavra
                coluna=1;
                linha=0;
            }else{
                printf("\n As coordenadas que escolheu nao sao validas, pois a palavra sai fora do tabuleiro!");
            }
        }else if (ori==1){
            if ((tamanho_palavra + linha)<=n+1){
                pontuacao(ori, coluna, tamanho_palavra, linha, palavra);
                criartab1(coluna, linha, n, tamanho_palavra, ori, palavra, coord);//Escrever a nova tabela com a palavra
                coluna=1;
                linha=0;
            }else{
                printf("\n As coordenadas que escolheu nao sao validas, pois a palavra sai fora do tabuleiro!");
            }
        }
    }
        printf("\n\n Caso queira parar de jogar introduza o numero '0' (caso contrario prima um inteiro sem ser o '0').");
        scanf("%d",&fim);
        if (fim==0){ //Quer voltar a jogar?(depois da jogada acabar).
            continue;
        }else{
            printf("\n Fez a escolha certa!");
            jogada++;
        }

    }
}

void menuajuda(){ // funcao auxiliar à opção h

    // Indica a funcionalidade de todas as opções de controlo.
    printf("\n");
    printf("-------------------- Menu Ajuda -----------------------\n");
    printf("-h ajuda para o utilizador\n");
    printf("-t lxc dimensoes do tabuleiro (linha x coluna)\n");
    printf("-d filename nome do ficheiro de dicionario a utilizar\n");
    printf("-l filename nome do ficheiro com letras a usar nas jogadas\n");
    printf("-m 5-20 numero de letras que um jogador pode ter na sua mao para jogar\n");
    printf("-n nn numero de jogadas maximo a realizar\n");
    printf("-i filename define ficheiro com o tabuleiro a usar em alternativa a jogar num tabuleiro vazio\n");
    printf("-j 1-4 modo de jogo 1 a 4\n");
    printf("-o filename define ficheiro onde escrever o tabuleiro final\n");
    printf("- r filename define ficheiro para registo de todas as jogadas possiveis\n");
    printf("\n");
}

int main(int argc, char *argv[]){

    //Valores 'default', isto é, quando nao sao escolhidas opcoes de comando
    static int linhas=9;
    int colunas, opcoes, jogo=1, jogadas, letrasnamao;
    char *dicionario;
    dicionario = "/usr/share/dict/words";

    //opcoes para as linhas de comando
         while((opcoes = getopt(argc, argv, "ht:d:lm:n:ij:or"))!= -1){
            switch(opcoes) {
                case 'h': //comando de ajuda ao utilizador
                    menuajuda();
                    return 0;
                    break;

                case 't'://comando que indica as dimensoes do tabuleiro
                    sscanf(optarg,"%dx%d", &linhas, &colunas);
                        if(linhas != colunas){
                            printf("\nPor favor insira valores iguais para as linhas e as colunas.\n\n");
                            return -1;
                        }
                        else if((linhas > 15 || linhas < 7) && (colunas > 15 || colunas < 7)) {
                            printf("\nO valor inserido nao pode ultrapassar o intervalo entre 7 e 15.\n\n");
                            return -1;
                        }
                        else if(linhas%2 == 0){
                            printf("\nO valor das linhas e das colunas tem de ser impar.\n\n");
                            return -1;
                        }
                        break;

                case 'd'://comando que indica o dicionario que vai ser usado
                    dicionario=optarg;
                    break;

                case 'l'://comando das letras na mao
                    printf("\n\nA opcao 'o' nao tem efeito nos modos de jogo implementados.\n\n");
                    break;

                case 'm'://comando do numero de letras na mao
                    letrasnamao = atoi(optarg);
                        if(letrasnamao > 20 || letrasnamao < 5){
                            printf("\nERRO! So pode ter entre 5 e 20 letras na mao!\n\n");
                            return -1;
                        }
                    printf("\nO numero de letras na mao e %d. No entando, a opcao 'm' nao tem efeito nos modos de jogo implementados.\n\n", letrasnamao);
                    break;

                case 'n'://comando do numero maximo de jogadas
                    jogadas = atoi(optarg);
                    printf("\nO numero maximo de jogadas e %d. No entando, a opcao 'n' nao tem efeito nos modos de jogo implementados.\n\n", jogadas);
                    break;

                case 'i'://comando da escolha do jogo
                    printf("\nA opcao 'i' nao tem efeito nos modos de jogo implementados.\n\n");
                    break;

                case 'j'://comando do modo de jogo
                    jogo = atoi(optarg);
                    break;

                case 'o'://comando do tabuleiro final
                    printf("\n\nA opcao 'o' nao tem efeito nos modos de jogo implementados.\n\n");
                    break;

                case 'r'://comando de registo das jogadas possiveis
                    printf("\n\nA opcao 'r' nao tem efeito nos modos de jogo implementados.\n\n");
                    break;
            }

        }
            if(jogo==1)
                mododejogo1(linhas);
            if(jogo==2)
                mododejogo2(linhas, dicionario);
            if(jogo == 3 || jogo == 4){
                printf("\nModo de jogo em desenvolvimento...\n\n"); //implica que os comandos: m,n,i,o,r deixem de ter significado visto que esses so se aplicam nos modos 3 e 4
                return -1;
            }if (jogo < 1 || jogo > 4){
                printf("\nERRO! So existem os modos de jogo 1,2,3 e 4!\n\n");// qualquer outro input dará erro
                return -1;
            }
}
