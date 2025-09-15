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

O método `check_solution(plan, verbose=False)` deve simular a execução do plano de ações passo a passo.  
O robot começa em `(0,0)` com capacidade de água `W0` e tempo `t=0`.  

Regras principais da simulação:

- Cada ação (`U, D, L, R, W`) consome exatamente **1 unidade de tempo**.
- **Movimentos (U, D, L, R)**:
  - Atualizam a posição segundo a direção.
  - Rejeitar se sair da grelha ou entrar num obstáculo.
  - Se após o movimento estiver em `(0,0)`, reabastece para `W0`.
- **Ação W (regar)**:
  - Só pode ser executada em células com plantas **não regadas ainda**.
  - Verificar se existe água suficiente (`water ≥ wk`).
  - Atualizar tempo; rejeitar se `time > dk` da planta.
  - Marcar célula como regada e descontar água.
- O plano deve usar apenas caracteres válidos `{U,D,L,R,W}`.
- No fim, **todas as plantas devem estar regadas exatamente uma vez**.
- O método retorna `True` se o plano for válido, `False` caso contrário.  
  (Se `verbose=True`, pode imprimir mensagens de erro mas continua a devolver só booleano.)

Nota: Não é necessário que o plano seja o mais curto; basta ser válido.
