# Cálculo da fronteira eficiente

Projeto de finanças em que é plotada a fronteira eficiente a partir da máximização do retorno e minimização de risco de um portifólio de ativos de renda variável

## Conceitos importantes

### Risco de carteira

O risco de uma carteira depende da forma como seus elementos se relacionam (covariam) entre si.

A redução do risco de uma carteira pode ser promovida pela seleção de ativos que mantenham relação inversa entre si.

Expressão geral de cálculo (Markowitz) do desvio-padrão de uma carteira de n ativos:

$$ \sigma_P = [\sum_{i=1}^N \sum_{j=1}^N w_i \cdot w_j \cdot \rho_{i,j} \cdot \sigma_i \cdot \sigma_j] ^{1/2} $$

Onde:

- $w_i$: percentual da carteira aplicado no ativo *i*;
- $\sigma_i$: desvio padrão dos retornos do ativo *i*;
- $\rho_{i,j}$: correlação entre os ativos *i* e *j*.

Ou no formato matricial:

$$ ( W \cdot COV \cdot W )^{1/2}$$

Lembrando da relação entre a covariância e a correlação é dada por:

$$ COV_{i,j} = \rho_{i,j} \cdot \sigma_i \cdot \sigma_j $$

O desvio padrão de uma carteira composta por três ativos A, B e C é calculado da seguinte forma:

$$ \sigma_P = [ (w_A^2 \cdot \sigma_A^2) + (w_B^2 \cdot \sigma_B^2) + (w_C^2 \cdot \sigma_C^2) + 2 \cdot w_A \cdot w_B \cdot Cov_{A,B} $$
$$ + 2 \cdot w_A \cdot w_C \cdot Cov_{A,C} + 2 \cdot w_B \cdot w_C \cdot Cov_{B,C}] ^{1/2}$$

## Fronteira eficiente

Na Fronteira Eficiente é possível selecionar uma carteira que apresenta, para determinado retorno, o menor risco possível. A escolha da melhor carteira é determinada pelo risco/retorno presente na avaliação de investimentos.
