# Métrica de Dificuldade de Palavras para Dislexia com K-means

Este projeto implementa um sistema inteligente para medir e classificar o nível de dificuldade de palavras sob a perspectiva de uma criança com dislexia. O sistema combina regras linguísticas com um algoritmo de Aprendizado de Máquina Não Supervisionado (K-Means) para agrupar as palavras de forma dinâmica em três níveis: Fácil, Médio e Difícil.

## Como o Sistema Funciona

Em vez de utilizar limites de corte manuais e estáticos, o algoritmo analisa múltiplos fatores estruturais e fonéticos das palavras para identificar padrões ocultos e agrupá-las organicamente:

1. **Complexidade Silábica:** Proporção do número de sílabas da palavra em relação ao tamanho máximo encontrado no conjunto.
2. **Similaridade Ortográfica:** Medida baseada na quantidade de derivações morfológicas da palavra.
3. **Letras Espelhadas:** Contagem de caracteres que frequentemente causam confusão visual ou rotação (b, d, p, q).
4. **Encontros Consonantais Complexos:** Identificação de dígrafos e junções fonéticas de alta carga cognitiva (ex: lh, nh, ch, pr, br, bl).

## Estrutura do Projeto

* `scikit-learn` + `StandardScaler`: Utilizados para normalizar as métricas, garantindo que características com escalas diferentes tenham o mesmo peso no modelo.
* `KMeans`: Executa o agrupamento (clusterização) iterativo e define o centro gravitacional de cada nível de dificuldade em tempo real.

## Como Executar o Teste

### Pré-requisitos
Certifique-se de ter o Python e as bibliotecas necessárias instaladas no seu ambiente:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python algoritmo-dislexia-kmeans/dislex_kmeans.py
