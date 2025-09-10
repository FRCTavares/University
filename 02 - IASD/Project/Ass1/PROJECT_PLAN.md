# IASD Trabalho #1 - Plano de Projecto Ultra Lean

**Equipa:** Francisco, Marta e Diogo
**Data de Início:** 9 de Setembro de 2025
**Prazo:** 26 de Setembro de 2025
**Deliverables:** `solution.py` implementando a classe `GardenerProblem`

---

## Resumo do Projecto

O Assignment #1 consiste em criar um programa que valide uma solução, não precisamos para já de criar o programa que chega a uma solução.

O robot vive numa grelha, começa em (0,0) com capacidade de água W0, e deve regar todas as plantas antes dos seus prazos.

A entrada vem de ficheiros `.dat` (definição do problema).

Os planos de saída são sequências de acções (`U, D, L, R, W`).

Temos de implementar uma classe `GardenerProblem` com dois métodos (Funções):

- **load(file)** → ler e analisar o ficheiro do problema, construir grelha/mapa, tipos de plantas e capacidade de água do robot.
- **check_solution(solution, verbose=False)** → simular a solução proposta passo a passo, verificando restrições:

- Apenas movimentos válidos (sem sair dos limites, sem entrar em células com obstáculos).
- Rega apenas em células com plantas, uma vez por célula.
- Respeitar prazos (`time ≤ dk`).
- O tanque de água deve ser suficiente; o reabastecimento acontece quando o robot volta a (0,0).
- Todas as plantas devem ser regadas exactamente uma vez.

- Retornar `True` se válido, `False` caso contrário.

**Não há necessidade de verificar se a solução é a melhor**. Qualquer plano válido passa.

---

## Roadmap (9 Set → 26 Set)

### Semana 1 (9–15 Set)

- Configurar pasta, colocar `solution.py`, `search.py`(Ficheiro dado), `utils.py`(Ficheiro dado), e `public1/` na mesma pasta.
- Francisco implementa análise de ficheiros em `load()`.
- Marta implementa simulação básica em `check_solution()`.
- Diogo configura executor de testes simples usando oficial `ex0.dat → ex0.plan`.
- Até ao fim da semana, devemos conseguir executar os primeiros testes públicos.
  
### Semana 2 (16–22 Set)

- Testar em todos os 10 casos públicos oficiais (`ex0`–`ex9`).
- Adicionar ou pensar em casos negativos rápidos: movimentos inválidos, regar duas vezes, prazos perdidos, etc...  ***Importante para o 20 fácil**
- Correcção de bugs e refinamento da análise e validação.
- Revisão de código e integração.

### Semana 3 (23–26 Set)
  
- Ter a certeza que está tudo fixe e verificações.
- Executar todos os testes novamente.
- Preparação da submissão: garantir que apenas entregamos o `solution.py`.
- Submeter em 26 Set antes do prazo.

---

## Distribuição de Tarefas

### Francisco – Análise de Entrada

- Implementar `load(file)`.
- Garantir que grelha, dimensões e tipos de plantas carregam correctamente.
- Tratar comentários `#`, linhas em branco e validação de formato.
- Fornecer estruturas de dados limpas para simulação.

### Marta – Simulação & Validação
  
- Implementar `check_solution(solution, verbose=False)`.
- Acompanhar posição do robot, tempo, água.
- Validar movimentos, regras de rega, prazos e reabastecimentos.
- Retornar `True` ou `False`, imprimir diagnósticos se `verbose=True`.
  
### Diogo – Testes
  
- Criar script pequeno para testar pares de `.dat` e `.plan`.
- Executar todos os casos de teste públicos (ex0–ex9).
- Criar testes de casos extremos: grelha vazia, caracteres inválidos, água insuficiente.
- Documentar falhas e apoiar correcção de bugs.

### Conjunto
  
- Acordar estruturas de dados simples (lista de grelha, dicionário de plantas, conjunto regado).
- Revisão de código do trabalho de cada um.
- Testes de integração e validação final.
- Submissão de `solution.py` único.
  
---

## Cenas Finais
  
- `solution.py` contém classe `GardenerProblem` com `load()` e `check_solution()` funcionais.
- Passa todos os 10 testes públicos.
- Rejeita planos inválidos correctamente.
- Código limpo e legível com comentários mínimos.
- Entregue antes do prazo.
