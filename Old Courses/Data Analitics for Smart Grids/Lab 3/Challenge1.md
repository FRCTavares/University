# **Seleção de Pseudo-Medições Ótimas para Estimação de Estado**

## **1. Introdução**
Neste trabalho, avaliamos o impacto da escolha de **apenas duas pseudo-medições** de um total de quatro disponíveis (\( S_1^m, S_2^m, S_3^m, S_4^m \)) na estimação de estado de um sistema elétrico. O objetivo é determinar quais as combinações que minimizam o erro total de estimação das tensões nos barramentos.

### **Objetivos**
- **Avaliar todas as combinações possíveis** de pseudo-medições e comparar os seus resultados.
- **Determinar a melhor escolha** com base no menor **erro RMS**.
- **Comparar o método reduzido** (apenas duas pseudo-medições) com a abordagem completa (quatro pseudo-medições).

---

## **2. Metodologia**
### **Passo 1: Modelo de Estimação**
A estimação de estado é baseada no método dos mínimos quadrados ponderados (WLS), dado por:

\[
\mathbit{x} = (\mathbit{H}_\mathbit{x}^T \mathbit{W} \mathbit{H}_\mathbit{x})^{-1} \mathbit{H}_\mathbit{x}^T \mathbit{W} \mathbit{z}
\]

Onde:
- \( \mathbit{x} \) representa o vetor de estado (tensões nos barramentos).
- \( \mathbit{H}_\mathbit{x} \) é a matriz de medições (adaptada para cada conjunto de pseudo-medições selecionado).
- \( \mathbit{W} \) é a matriz de pesos (atribuindo maior peso a medições reais).
- \( \mathbit{z} \) é o vetor de medições, incluindo as pseudo-medições escolhidas.

### **Passo 2: Testar Todas as Combinações**
Foram avaliadas as seguintes **seis combinações** de pseudo-medições:

1. \( (S_1^m, S_2^m) \)
2. \( (S_1^m, S_3^m) \)
3. \( (S_1^m, S_4^m) \)
4. \( (S_2^m, S_3^m) \)
5. \( (S_2^m, S_4^m) \)
6. \( (S_3^m, S_4^m) \)

Cada combinação foi testada e analisada com base em:
- **Erro RMS das tensões estimadas**.
- **Precisão da corrente relativa** (\( I_{12} \) e \( I_{54} \)).
- **Estabilidade dos resultados ao longo de múltiplas execuções**.

---

## **3. Resultados e Análise**
### **Comparação dos Erros RMS**
Os erros RMS totais para cada combinação foram os seguintes:

| Combinação de Pseudo-Medições | Erro RMS Total |
|------------------------------|---------------|
| \( (S_1^m, S_2^m) \) | 0.05966 |
| \( (S_1^m, S_3^m) \) | 0.03044 |
| \( (S_1^m, S_4^m) \) | 0.05966 |
| \( (S_2^m, S_3^m) \) | 0.03042 |
| \( (S_2^m, S_4^m) \) | 0.05930 |
| **\( (S_3^m, S_4^m) \)** | **0.02942** (Melhor opção) |

### **Melhor Combinação**
A combinação que apresentou **menor erro total** foi **\( (S_3^m, S_4^m) \)**, garantindo a melhor precisão na estimação das tensões, mantendo os valores mais próximos dos esperados teoricamente.

---

## **4. Comparação com a Abordagem Completa**
Para validar a escolha, comparamos o método de **duas pseudo-medições** com a **abordagem completa** (quatro pseudo-medições).

- **Diferenças nas estimativas de tensão** foram analisadas.
- **Erros percentuais** foram calculados para cada barramento.
- **O impacto da redução de pseudo-medições** foi avaliado.

Os resultados indicaram que **a abordagem reduzida** introduz apenas **pequenas variações** nos valores das tensões estimadas, enquanto reduz a complexidade computacional.

---

## **5. Conclusão**
### **Principais Conclusões**
- A seleção **criteriosa** de pseudo-medições pode garantir um **desempenho próximo da abordagem completa**, reduzindo a necessidade de medições adicionais.
- A combinação **\( (S_3^m, S_4^m) \)** foi a mais eficaz, resultando no **menor erro RMS**.
- **Menos pseudo-medições podem ser utilizadas sem comprometer significativamente a precisão da estimação**.

### **Trabalhos Futuros**
- Aplicação deste método em **redes elétricas maiores**.
- Desenvolvimento de um **algoritmo adaptativo** que selecione as melhores pseudo-medições em tempo real.

---

## **6. Figuras e Visualizações**
Inclui:
- **Gráficos comparativos** das combinações de pseudo-medições.
- **Diferenças de tensão entre a abordagem completa e a abordagem reduzida**.
- **Análise de erros percentuais** nas tensões estimadas.

---

Este documento apresenta uma **análise estruturada** da escolha ótima de pseudo-medições, garantindo um equilíbrio entre precisão e eficiência computacional.
