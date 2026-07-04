# Classificação de Palavras para Dislexia com K-Means

Sistema que classifica palavras em níveis de dificuldade (Fácil, Médio, Difícil) para auxiliar crianças com dislexia no processo de alfabetização. O projeto combina métricas linguísticas com o algoritmo K-Means.

## Métricas

- Frequência de Uso
- Complexidade Silábica
- Similaridade Ortográfica
- Encontros Consonantais Complexos

## Cálculo do Nível

Se SO == 0:
Nível = (0.30 * Frequência) + (0.70 * Complexidade Silábica)
Senão:
Nível = (0.20 * Frequência) + (0.40 * Complexidade Silábica) + (0.40 * Similaridade Ortográfica)


## Classificação

- Fácil: Nível <= 0.3
- Médio: 0.3 < Nível <= 0.6
- Difícil: Nível > 0.6

## Como Executar

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iPYIVsKPBV2enDwXeaF6w19VmqUmJuZ9?usp=sharing)

### Localmente

```bash
python -m venv venv
pip install -r requirements.txt
python dislex_kmeans.py