# Desafio: Reduzir a Dimensionalidade de \(X\) Utilizando Apenas os Termos Quadráticos

No documento “2.2 Losses Prediction”, uma das propostas para simplificar a construção da matriz \(X\) na equação

\[
y = X\,\beta + \varepsilon
\]

consiste em **manter apenas os termos quadráticos** das injeções de potência, ignorando os produtos cruzados \(2\,P_i\,P_j\). Isto reduz significativamente o número de colunas em \(X\) quando comparado à expansão completa (que inclui todas as combinações \(P_i^2\) e \(2\,P_i\,P_j\)).

## 1. Conceito

Em vez de termos:

\[
X =
\begin{bmatrix}
P_1^2 & 2P_1P_2 & 2P_1P_3 & \cdots & P_2^2 & 2P_2P_3 & \cdots
\end{bmatrix},
\]

passamos a usar apenas:

\[
X_{\text{sq}} =
\begin{bmatrix}
P_1^2 & P_2^2 & \cdots & P_n^2
\end{bmatrix}.
\]

Deste modo, cada coluna de \(X_{\text{sq}}\) corresponde ao quadrado de uma injeção \(P_i\). O objetivo é **reduzir a dimensionalidade** e, assim, **simplificar** o modelo.

## 2. Construção da Nova Matriz \(X_{\text{sq}}\)

Supondo que:
- \(P\) é a matriz de injeções de potência, com dimensão \((M, n)\), onde \(M\) é o número de instantes e \(n\) é o número de buses não slack.
- Para cada instante \(k\), temos as injeções \(\bigl(P_{k,1}, P_{k,2}, \dots, P_{k,n}\bigr)\).

A nova matriz \(X_{\text{sq}}\) fica com dimensão \((M, n)\), onde cada elemento é:

\[
X_{\text{sq}}(k, i) = P_{k, i}^2.
\]

### Exemplo de Código (Python)

```python
import numpy as np

M, n = P.shape  # M instantes, n buses
X_sq = np.zeros((M, n))

for k in range(M):
    for i in range(n):
        X_sq[k, i] = P[k, i]**2