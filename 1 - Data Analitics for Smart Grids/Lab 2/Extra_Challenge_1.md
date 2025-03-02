# Desafio: Modelo de Perdas para Redes de Baixa Tensão

Neste desafio, é necessário repensar a equação (16), que originalmente tem a forma:

$$
y = X\\beta + \\varepsilon,
$$

num contexto em que a rede é de baixa tensão (LV). Em redes LV, assume-se que as diferenças de ângulo entre os buses são negligíveis, sendo as variações nas magnitudes de tensão os fatores dominantes. Adicionalmente, assume-se que os elementos de susceptância \(B_{ij}\) são muito inferiores aos de condutância \(G_{ij}\).

## Abordagem Proposta

### 1. Simplificação do Modelo

- **Diferenças de Ângulo Negligíveis:**  
  Em redes de baixa tensão, os ângulos dos buses são praticamente iguais, pelo que se pode ignorar o seu efeito na determinação das perdas.

- **Foco nas Diferenças de Tensão:**  
  As perdas numa linha que liga o bus \(i\) ao bus \(j\) podem ser aproximadas por:

  $$
  P_{loss}^{(ij)} \\approx G_{ij} \\,(V_i - V_j)^2,
  $$

  onde \(V_i\) e \(V_j\) são as magnitudes de tensão nos buses \(i\) e \(j\).

### 2. Construção da Matriz de Características \(X'\)

Em vez de utilizar a matriz \(X\) tradicional com termos polinomiais dos injetados (por exemplo, \(P_i^2\) e \(2P_iP_j\)), propõe-se construir uma nova matriz \(X'\) em que:
- Cada coluna corresponde a uma linha da rede.
- Cada elemento é calculado como \((V_i - V_j)^2\) para a ligação que une os buses \(i\) e \(j\) no instante de tempo considerado.

### 3. Modelo de Regressão

O novo modelo assume a forma:

$$
y = X' \\beta' + \\varepsilon,
$$

onde:
- \(y\) é o vetor das perdas (obtidas, por exemplo, através de medições ou de um modelo físico),
- \(X'\) é a nova matriz de características baseada nas diferenças de tensão,
- \(\beta'\) é o vetor dos parâmetros a estimar, que idealmente deverá aproximar os valores de \(G_{ij}\) (ou uma versão escalada destes).

A estimação dos parâmetros pode ser efetuada através do método dos Mínimos Quadrados Ordinários (OLS):

$$
\\beta' = \\bigl((X')^T X'\\bigr)^{-1} (X')^T y,
$$

e a predição é obtida por:

$$
\\hat{y} = X' \\beta'.
$$

## Conclusão

Esta abordagem adapta o modelo de regressão original para redes de baixa tensão, onde as perdas são dominadas pelas pequenas variações nas magnitudes das tensões, e não pelos ângulos. Assim, o modelo torna-se mais adequado para redes LV, refletindo a realidade em que \(B_{ij} \\ll G_{ij}\).

Esta solução permite explorar a influência da topologia da rede e das características de baixa tensão no comportamento das perdas, oferecendo uma alternativa interessante ao método original.
