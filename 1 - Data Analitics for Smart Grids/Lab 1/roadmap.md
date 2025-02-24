# Lab Report Roadmap

## Problema 1: Identificação de Fases (Phase Identification)
### 1.1 Introdução
- Contextualização do problema e definição dos objetivos.
- Descrição dos dados:
  - Número de consumidores (N)
  - Períodos de análise (M)
  - Ruído (valor do desvio padrão, ex.: 0.10)
- Justificação da metodologia adotada.

### 1.2 Como Resolvemos:
- **Construção das Matrizes:**
  - Matriz X: Extração dos dados no intervalo de tempo definido.
  - Matriz Y: Agrupamento dos dados conforme as fases atribuídas.
- **Estimativa da Matriz β:**
  - Estimação via mínimos quadrados.
  - Conversão para valores binários e obtenção da fase estimada.

### 1.3 Resultados e Análise
- **Gráficos:**
  - [Gráfico das barras] - verificação e confirmação das legendas.
  - [Gráfico dos valores de β] - visualização dos resultados Confusion Matrix.
- **Discussão dos Resultados:**
  - Comentários e interpretação dos gráficos.
  - Análise da acurácia e dos efeitos do ruído.

## Desafios Extra

### 2. Número de Clientes Superior a 4
- **Objetivo:** Analisar o impacto do aumento do número de consumidores (N) em relação ao número de observações (M).
- **Subitens:**
  - [Gráficos para N = 6, M = 12]
  - [Gráficos para N = 11, M = 12]
- **Análise:**
  - Percentagem de classificações corretas à medida que N varia.
  - Discussão sobre a robustez do modelo com diferentes relações entre M e N.

### 3. Dois Clientes com o Mesmo Consumo
- **Objetivo:** Verificar o comportamento do modelo quando dois clientes apresentam exatamente o mesmo consumo.
- **Subitens:**
  - Análise de casos com consumo idêntico.
  - [Gráfico dos resultados para clientes com consumo idêntico]
  - Discussão sobre a sensibilidade e estabilidade do modelo.

#### 3.1 Dois Clientes com Consumo Muito Semelhante
- **Objetivo:** Analisar o caso em que os consumos são quase idênticos.
- **Subitens:**
  - Representação gráfica dos resultados.
  - Comentários sobre a diferença (ou falta dela) e a robustez do modelo.

### 4. Clientes com 3 Fases
- **Objetivo:** Adaptar a metodologia para lidar com clientes trifásicos.
- **Subitens:**
  - Desenvolvimento e implementação de um método para 3 fases.
  - [Gráficos de clusterização e PCA para clientes com 3 fases]
  - Discussão dos desafios e das adaptações necessárias.

---

## Conclusão
- Resumo dos principais resultados.
- Discussão sobre a robustez do modelo e a sensibilidade em diferentes cenários.
- Sugestões e recomendações para trabalhos futuros.

---

## Referências
- Lista de fontes bibliográficas.
- Links para os ficheiros de dados e documentação.
