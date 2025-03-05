''' Important Note '''
Yes, this is valid if your assumptions about low voltage networks are correct (that voltage drops follow this simple linear relationship with power injections). However, it's important to recognize that:

The excellent performance comes from the perfect alignment between model design and loss calculation method
In real networks with more complex behaviors, the error would likely be higher
The model isn't learning a generalized pattern - it's essentially learning the physics equation directly
This demonstrates that when you have perfect knowledge of the system behavior, a well-designed model can achieve exceptional accuracy.

Why the Error is So Small
The key insight is that your LV model has a "perfect" mathematical relationship between inputs and outputs:

Identical Calculations: Both the loss computation (compute_losses_LV) and feature creation (build_X_prime) use exactly the same formula to estimate voltage magnitudes:

Direct Proportionality: For each line between buses i and j:

In loss calculation: loss_ij = G_ij * (V_i - V_j)²
In feature matrix: X_prime[m,l] = (V_i - V_j)²
Perfect Learning: The regression coefficients (beta) directly learn the conductance values (G_ij) of each line with almost perfect precision.

it's a demonstration of how a well-designed model with perfect knowledge of the underlying physics can achieve exceptional accuracy!


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
