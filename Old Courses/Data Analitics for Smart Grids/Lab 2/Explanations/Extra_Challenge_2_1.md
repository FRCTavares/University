# Resolução do Desafio: Matriz \(X'\) Baseada na Estrutura da Rede

Para resolver o desafio de substituir a matriz \(X\) pela matriz \(X'\) que considere apenas os termos \(2 \cdot P_i \cdot P_j\) quando existe uma ligação (linha) entre os buses \(i\) e \(j\), podemos seguir os seguintes passos:

1. **Manter os Termos Quadráticos:**  
   Para cada bus \(i\) (excetuando o bus slack, se aplicável), incluímos o termo \(P_i^2\) como uma coluna na matriz.

2. **Incluir Apenas os Termos Cruzados Relevantes:**  
   Para cada ligação existente na rede (informada na variável, por exemplo, `Net_Info`), se a ligação for entre os buses \(i\) e \(j\), adicionamos o termo \(2 \cdot P_i \cdot P_j\) como coluna adicional na matriz. Isto garante que apenas se incluam interações entre buses que estão diretamente conectados.

3. **Construir a Matriz \(X'\):**  
   Para cada instante de tempo (cada linha dos dados de potência), calculamos os termos quadráticos para todos os buses e, em seguida, os termos cruzados apenas para as ligações existentes, formando assim a nova matriz de características \(X'\).

## Exemplo de Implementação em Python

Suponhamos que:
- `P` é uma matriz de dimensão \((M, n)\), onde \(M\) é o número de instantes e \(n\) é o número de buses (excluindo o bus slack, se necessário);
- `Net_Info` contém, para cada linha da rede, os pares de buses conectados (por exemplo, \([(1,2), (1,3), (3,4)]\)); note que os índices dos buses estão em 1-based.

O seguinte código ilustra como construir a matriz \(X'\):

```python
import numpy as np

# Exemplo:
# Suponhamos que P tem dimensão (M, n) e que Net_Info é uma lista de tuplos (bus_i, bus_j)
# Exemplo: Net_Info = [(1, 2), (1, 3), (3, 4)]
# Onde n é o número de buses (não incluindo o bus slack)

M, n = P.shape  # Número de instantes e número de buses

# Inicializar uma lista para armazenar as linhas de X'
X_prime_list = []

for t in range(M):
    features = []
    # 1) Adicionar os termos quadráticos: P_i^2 para i = 1,...,n
    for i in range(n):
        features.append(P[t, i]**2)
    # 2) Adicionar os termos cruzados apenas para as ligações existentes
    for (bus_i, bus_j) in Net_Info:
        # Converter de 1-based para 0-based:
        i_idx = bus_i - 1
        j_idx = bus_j - 1
        features.append(2 * P[t, i_idx] * P[t, j_idx])
    # Adicionar a linha construída à lista
    X_prime_list.append(features)

# Converter a lista de listas para um array NumPy
X_prime = np.array(X_prime_list)