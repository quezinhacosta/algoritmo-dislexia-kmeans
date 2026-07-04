import time
import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

plt.style.use('dark_background')

try:
    nlp = spacy.load("pt_core_news_sm")
except:
    import os
    os.system("python -m spacy download pt_core_news_sm")
    nlp = spacy.load("pt_core_news_sm")

palavras_teste = [
    "casa", "bola", "dado", "pato", "sapato", "janela", "cadeira",
    "problema", "palhaço", "cachorro", "passarinho", "subtração", "perspectiva",
    "braço", "quadro", "planta", "bota", "gato", "psicologia", "ritmo"
]

dados_base = {
    "Palavra": palavras_teste,
    "Numero_de_silabas": [2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 3, 4, 2, 2, 2, 2, 2, 5, 2],
    "Derivações": [5, 4, 2, 1, 3, 2, 2, 1, 3, 2, 1, 0, 0, 3, 2, 2, 1, 2, 1, 0]
}
df_mock = pd.DataFrame(dados_base)

def buscar_linha(palavra):
    return df_mock[df_mock["Palavra"].str.lower() == str(palavra).lower()]

def complexidade_silabica(palavra):
    linha = buscar_linha(palavra)
    if linha.empty: return 0
    silabas = linha["Numero_de_silabas"].values[0]
    max_silabas = df_mock["Numero_de_silabas"].max()
    return silabas / max_silabas if max_silabas != 0 else 0

def similaridade_ortografica(palavra):
    linha = buscar_linha(palavra)
    if linha.empty: return 0
    derivada_palavra = linha["Derivações"].values[0]
    maior_derivada = df_mock["Derivações"].max()
    return derivada_palavra / maior_derivada if maior_derivada != 0 else 0

def contagem_letras_espelhadas(palavra):
    palavra = palavra.lower()
    letras_conflitantes = ['b', 'd', 'p', 'q', 'd', 'b']
    total = sum(1 for letra in palavra if letra in letras_conflitantes)
    return total

def contagem_encontros_complexos(palavra):
    palavra = palavra.lower()
    padroes_complexos = ['lh', 'nh', 'ch', 'rr', 'ss', 'tr', 'pr', 'br', 'cr', 'vr', 'fl', 'cl', 'pl', 'ps']
    total = sum(1 for padrao in padroes_complexos if padrao in palavra)
    return total

print("Calculando métricas estruturais e fonéticas...")
X_list = []
for palavra in palavras_teste:
    CS = complexidade_silabica(palavra)
    SO = similaridade_ortografica(palavra)
    LE = contagem_letras_espelhadas(palavra)
    EC = contagem_encontros_complexos(palavra)
    X_list.append([CS, SO, LE, EC])

X = np.array(X_list)

scaler = StandardScaler()
X_escalado = scaler.fit_transform(X)

print("Iniciando representação em tempo real do K-Means...")
time.sleep(1.0)

# CORES DEFINIDAS PARA CADA CLUSTER
cores_clusters = ['#FF6B6B', '#4ECDC4', '#FFD93D']  # Vermelho, Verde-água, Amarelo
nomes_clusters = ['Cluster 1', 'Cluster 2', 'Cluster 3']

# ========== IMPLEMENTAÇÃO MANUAL DO K-MEANS ==========

np.random.seed(42)
n_clusters = 3
n_features = X_escalado.shape[1]

# Escolhe 3 pontos ALEATÓRIOS do dataset como centroides iniciais
indices_aleatorios = np.random.choice(len(X_escalado), n_clusters, replace=False)
centroides = X_escalado[indices_aleatorios].copy()

print("Centroides iniciais escolhidos ALEATORIAMENTE entre os pontos de dados!")
time.sleep(1.0)

# Guardar o gráfico final para exibir depois
grafico_final = None
labels_final = None
centroides_final = None

# Loop de iterações do K-Means (máximo 10 iterações)
for iteracao in range(1, 11):
    # PASSO 1: Atribuir cada ponto ao centroide mais próximo
    distancias = []
    for ponto in X_escalado:
        dist_ao_centroide = [np.linalg.norm(ponto - centroide) for centroide in centroides]
        distancias.append(dist_ao_centroide)
    
    distancias = np.array(distancias)
    labels = np.argmin(distancias, axis=1)
    
    # PASSO 2: Calcular novos centroides
    novos_centroides = []
    for k in range(n_clusters):
        pontos_do_cluster = X_escalado[labels == k]
        if len(pontos_do_cluster) > 0:
            novo_centroide = np.mean(pontos_do_cluster, axis=0)
        else:
            novo_centroide = X_escalado[np.random.choice(len(X_escalado))]
        novos_centroides.append(novo_centroide)
    
    novos_centroides = np.array(novos_centroides)
    
    # Verificar convergência
    convergiu = np.allclose(centroides, novos_centroides, atol=1e-4)
    
    # Atualizar centroides
    centroides = novos_centroides.copy()
    
    # ========== VISUALIZAÇÃO ==========
    plt.figure(figsize=(12, 7), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')
    
    # CORES DOS PONTOS
    cores_pontos = [cores_clusters[label] for label in labels]
    
    # PLOTAR PONTOS
    scatter = plt.scatter(X[:, 0], X[:, 3], 
                         c=cores_pontos,
                         s=120, 
                         alpha=0.8, 
                         edgecolors='white', 
                         linewidth=0.5)
    
    # PLOTAR CENTROIDES
    centroides_reais = scaler.inverse_transform(centroides)
    for j, centroide in enumerate(centroides_reais):
        tamanho_cluster = np.sum(labels == j)
        plt.scatter(centroide[0], centroide[1],
                   c=cores_clusters[j],
                   marker='X', 
                   s=400, 
                   edgecolors='white', 
                   linewidth=3,
                   label=f'Centroide {j+1} (n={tamanho_cluster})')
    
    # DESTACAR PONTOS QUE MUDARAM
    if iteracao > 1:
        mudaram = labels != labels_anterior
        if np.any(mudaram):
            pontos_mudaram = X[mudaram]
            plt.scatter(pontos_mudaram[:, 0], pontos_mudaram[:, 3],
                       facecolors='none',
                       edgecolors='yellow',
                       s=180,
                       linewidth=2,
                       label=f'Pontos que mudaram ({np.sum(mudaram)})')
    
    # CONFIGURAÇÕES
    titulo = f"Iteração {iteracao}"
    if convergiu:
        titulo += " - CONVERGIU! "
    
    plt.title(titulo, fontsize=16, fontweight='bold', color='white')
    plt.xlabel("Complexidade Silábica (Proporcional)", fontsize=12, color='white')
    plt.ylabel("Encontros Consonantais Complexos (Contagem)", fontsize=12, color='white')
    
    plt.xticks(color='white')
    plt.yticks(color='white')
    
    legend = plt.legend(facecolor='black', edgecolor='white', loc='upper left')
    for text in legend.get_texts():
        text.set_color('white')
    
    plt.grid(True, linestyle='--', alpha=0.2, color='gray')
    plt.tight_layout()
    
    # Mostrar o gráfico
    plt.show()
    
    # INFORMAÇÕES DA ITERAÇÃO
    print(f"\nIteração {iteracao}:")
    for k in range(n_clusters):
        n_pontos = np.sum(labels == k)
        print(f"  Cluster {k+1}: {n_pontos} palavras")
    
    # Mostrar distribuição
    print("\nDistribuição atual:")
    for k in range(n_clusters):
        palavras_cluster = [palavras_teste[i] for i in range(len(palavras_teste)) if labels[i] == k]
        if len(palavras_cluster) > 0:
            print(f"  {cores_clusters[k]} Cluster {k+1}: {', '.join(palavras_cluster)}")
    
    # Guardar para o gráfico final
    if iteracao == 10 or convergiu:
        labels_final = labels.copy()
        centroides_final = centroides.copy()
    
    if convergiu:
        print("\n Algoritmo convergiu! Centroides estabilizados.")
        # Aguardar um pouco para ver o gráfico final
        time.sleep(2.0)
        break
    
    # Guardar labels para comparar
    labels_anterior = labels.copy()
    
    # Pausa para visualização
    if iteracao < 10:
        print("\n" + "="*60)
        print("" Próxima iteração em 3 segundos...")
        time.sleep(3.0)
        clear_output(wait=True)  # Limpa APENAS para a próxima iteração

# ========== LIMPAR TELA FINAL E MOSTRAR RESULTADOS ==========
clear_output(wait=True)

print("="*60)
print("ALGORITMO DE AGRUPAMENTO FINALIZADO!")
print("="*60)


if labels_final is not None:
    plt.figure(figsize=(12, 7), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')
    
    cores_pontos_final = [cores_clusters[label] for label in labels_final]
    
    scatter = plt.scatter(X[:, 0], X[:, 3], 
                         c=cores_pontos_final,
                         s=150, 
                         alpha=0.9, 
                         edgecolors='white', 
                         linewidth=1)
    
    centroides_reais_final = scaler.inverse_transform(centroides_final)
    for j, centroide in enumerate(centroides_reais_final):
        tamanho_cluster = np.sum(labels_final == j)
        plt.scatter(centroide[0], centroide[1],
                   c=cores_clusters[j],
                   marker='X', 
                   s=500, 
                   edgecolors='white', 
                   linewidth=4,
                   label=f'Centroide {j+1} (n={tamanho_cluster})')
    
    plt.title("RESULTADO FINAL - K-MEANS CONVERGIDO ", fontsize=16, fontweight='bold', color='white')
    plt.xlabel("Complexidade Silábica (Proporcional)", fontsize=12, color='white')
    plt.ylabel("Encontros Consonantais Complexos (Contagem)", fontsize=12, color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')
    
    legend = plt.legend(facecolor='black', edgecolor='white', loc='upper left')
    for text in legend.get_texts():
        text.set_color('white')
    
    plt.grid(True, linestyle='--', alpha=0.2, color='gray')
    plt.tight_layout()
    plt.show()
else:

    labels_final = labels.copy()
    centroides_final = centroides.copy()

df_resultado = pd.DataFrame(palavras_teste, columns=['Palavra'])
df_resultado['Cluster'] = labels_final
df_resultado['Letras_Espelhadas'] = X[:, 2]
df_resultado['Encontros_Complexos'] = X[:, 3]

ranking_clusters = df_resultado.groupby('Cluster')['Encontros_Complexos'].mean().sort_values().index
mapeamento_niveis = {
    ranking_clusters[0]: "facil",
    ranking_clusters[1]: "medio",
    ranking_clusters[2]: "dificil"
}
df_resultado['Classificação'] = df_resultado['Cluster'].map(mapeamento_niveis)
df_resultado = df_resultado.drop(columns=["Cluster"])

print("\n TABELA DE CLASSIFICAÇÃO FINAL:")
print("="*60)
print(df_resultado.sort_values(by='Classificação').to_string(index=False))

print("\nRESUMO POR NÍVEL:")
print("="*60)
for nivel in ['facil', 'medio', 'dificil']:
    palavras_nivel = df_resultado[df_resultado['Classificação'] == nivel]['Palavra'].tolist()
    cores_nivel = {
        'facil': '🟢',
        'medio': '🟡',
        'dificil': '🔴'
    }
    print(f"{cores_nivel[nivel]} {nivel.upper()}: {', '.join(palavras_nivel)}")

print("\n" + "="*60)
print("Análise concluída com sucesso!")