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

## Cenas Finais
  
- `solution.py` contém classe `GardenerProblem` com `load()` e `check_solution()` funcionais.
- Passa todos os 10 testes públicos.
- Rejeita planos inválidos correctamente.
- Código limpo e legível com comentários mínimos.
- Entregue antes do prazo.

## Resumo da Marta
