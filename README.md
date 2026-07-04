
A classificação é então determinada por:
- **Fácil:** Nível <= 0.3
- **Médio:** 0.3 < Nível <= 0.6
- **Difícil:** Nível > 0.6

### Inicialização Inteligente dos Centroides

Diferente da abordagem tradicional de K-Means que usa centroides aleatórios, este sistema utiliza uma inicialização inteligente:

1. **Centroides são posicionados com base na classificação calculada** das palavras
2. **Ajuste fino do centroide "Difícil"** para capturar palavras com alto índice de encontros consonantais complexos
3. **Distância mínima entre centroides** para evitar convergência para mínimos locais

### Otimização por Match

O algoritmo executa até 15 iterações e monitora continuamente o **match** entre:
- A classificação do K-Means (agrupamento)
- A classificação calculada (baseada na fórmula original)

A melhor configuração de centroides é preservada, garantindo o melhor alinhamento possível entre as duas classificações.

## Estrutura do Código

### Componentes Principais

- **`estimar_silabas()`**: Estima o número de sílabas usando regras fonéticas do português
- **`estimar_derivacoes()`**: Estima derivações baseado em sufixos e padrões morfológicos
- **`contagem_encontros_complexos()`**: Conta padrões consonantais complexos
- **`frequencia_de_uso()`**: Retorna a frequência normalizada da palavra
- **`classificar_nivel()`**: Classifica a palavra baseado no nível calculado
- **`calcular_match()`**: Calcula o alinhamento entre K-Means e classificação calculada
- **`criar_grafico()`**: Gera visualizações interativas do processo de clusterização

### Visualização

O sistema gera gráficos em tempo real mostrando:
- **Pontos coloridos** representando cada palavra, com cor baseada no cluster atribuído
- **Centroides (X)** representando o centro de cada cluster
- **Match percentage** atualizado a cada iteração
- **Pontos destacados** que mudaram de cluster entre iterações

### Eixos do Gráfico

- **Eixo X:** Frequência de Uso (Normalizada)
- **Eixo Y:** Encontros Consonantais Complexos

Os eixos são configurados para exibir a faixa 0-1 com margem de 15%, facilitando a visualização dos agrupamentos.

## Dataset

O sistema utiliza um dataset de **135 palavras** em português, com frequências de uso extraídas de corpora linguísticos. As palavras variam desde termos simples como "flor" e "sol" até palavras complexas como "aeroporto" e "aprendizagem".

## Pré-requisitos

Certifique-se de ter o Python e as bibliotecas necessárias instaladas no seu ambiente:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
python dislex_kmeans.py
# ou
venv\Scripts\activate  # Windows
pip install numpy pandas matplotlib scikit-learn
python dislex_kmeans.py

Execute o notebook diretamente no seu navegador: 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iPYIVsKPBV2enDwXeaF6w19VmqUmJuZ9?usp=sharing)