import time
import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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
    # Mede a presença de caracteres que frequentemente sofrem rotação ou inversão visual
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


for i in range(1, 8):
    clear_output(wait=True)
    

    kmeans = KMeans(n_clusters=3, max_iter=i, init='random', n_init=1, random_state=15)
    labels = kmeans.fit_predict(X_escalado)
    centroides = kmeans.cluster_centers_
  
    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(X[:, 0], X[:, 3], c=labels, cmap='plasma', s=80, alpha=0.8, edgecolors='k')
    
    centroides_reais = scaler.inverse_transform(centroides)
    plt.scatter(centroides_reais[:, 0], centroides_reais[:, 3], c='cyan', marker='X', s=250, label='Centros de Dificuldade')
    
    plt.title(f"Ajuste Dinâmico dos Níveis - Iteração {i}", fontsize=14, fontweight='bold')
    plt.xlabel("Complexidade Silábica (Proporcional)", fontsize=12)
    plt.ylabel("Encontros Consonantais Complexos (Contagem)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    
    print(f"Iteração número {i} processada.")
    print("Os centros de gravidade estão se movendo para equilibrar os grupos de palavras.")
    time.sleep(1.2)

clear_output(wait=True)
print("Algoritmo de agrupamento finalizado.")

df_resultado = pd.DataFrame(palavras_teste, columns=['Palavra'])
df_resultado['Cluster'] = labels
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

print("\nTabela de classificação resultante do teste:")
print(df_resultado.sort_values(by='Classificação'))
