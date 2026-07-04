import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.preprocessing import StandardScaler
import unicodedata

plt.ion()
plt.style.use('dark_background')

# ============================================================================
# DADOS EMBUTIDOS
# ============================================================================

palavras_teste = [
    "aeroporto", "almoço", "aniversário", "aprendizagem", "aquarela",
    "bicicleta", "biscoito", "borboleta", "brigadeiro", "brincadeira",
    "chegando", "churrasco", "correndo", "dente", "elefante",
    "escova", "gostava", "homem", "jantou", "machucado",
    "presente", "ajuda", "amanhã", "bicho", "bruxa",
    "chapéu", "chave", "cinco", "chuva", "dedo",
    "desenhe", "diferença", "disse", "doce", "dois",
    "durante", "escola", "explique", "foguete", "jardim",
    "janela", "junho", "menino", "flor", "panela",
    "queria", "sol", "tinta", "trabalho", "bilhete",
    "explicação", "facil", "garrafa", "inimigo", "lentamente",
    "lingua", "mexer", "acabou", "assusta", "caíram",
    "chamavam", "correr", "desça", "heroi", "nadaram",
    "ônibus", "seguia", "vermelho", "diferente", "fechar",
    "frente", "natureza", "nenhum", "plantas", "pomba",
    "príncipe", "pulou", "salva", "acordou", "beleza",
    "bolacha", "comprou", "pegaram", "salvar", "soltaram",
    "soltou", "conhecer", "fumaça", "geladeira", "lâmpada",
    "mulher", "papel", "perguntou", "sentiu", "sinal",
    "combinar", "criança", "estavam", "folhas", "pessoa",
    "convidar", "pegou", "descobrir", "passeando", "passou",
    "próximo", "aconteceu", "chegar", "ensinar", "ficaram",
    "seguir", "assustar", "começou", "tarde", "limpar",
    "pequena", "pequeno", "assustado", "cantar", "policial",
    "subiu", "feliz", "caçador", "sujo", "alguém",
    "esquece", "pegue", "quatro", "quero", "minha",
    "muitos", "nesse", "ninguém", "pássaro", "português"
]

frequencias = {
    "aeroporto": 2724, "almoço": 6, "aniversário": 101, "aprendizagem": 106, "aquarela": 18,
    "bicicleta": 1492, "biscoito": 610, "borboleta": 473, "brigadeiro": 82, "brincadeira": 3681,
    "chegando": 5029, "churrasco": 512, "correndo": 3512, "dente": 1308, "elefante": 768,
    "escova": 462, "gostava": 3871, "homem": 65326, "jantou": 132, "machucado": 871,
    "presente": 8148, "ajuda": 17961, "amanhã": 477, "bicho": 715, "bruxa": 1535,
    "chapéu": 31, "chave": 5532, "cinco": 15165, "chuva": 3315, "dedo": 2477,
    "desenhe": 111, "diferença": 22, "disse": 112706, "doce": 5568, "dois": 55808,
    "durante": 14064, "escola": 15354, "explique": 835, "foguete": 664, "jardim": 3215,
    "janela": 5142, "junho": 1111, "menino": 8888, "flor": 1792, "panela": 364,
    "queria": 40201, "sol": 1221, "tinta": 1053, "trabalho": 45159, "bilhete": 1752,
    "explicação": 3, "facil": 159, "garrafa": 2863, "inimigo": 4997, "lentamente": 1296,
    "lingua": 177, "mexer": 2174, "acabou": 16203, "assusta": 1182, "caíram": 103,
    "chamavam": 499, "correr": 5496, "desça": 20, "heroi": 105, "nadaram": 17,
    "ônibus": 95, "seguia": 273, "vermelho": 4423, "diferente": 11509, "fechar": 3200,
    "frente": 26891, "natureza": 4071, "nenhum": 25256, "plantas": 1363, "pomba": 208,
    "príncipe": 177, "pulou": 573, "salva": 1115, "acordou": 1195, "beleza": 5971,
    "bolacha": 99, "comprou": 2672, "pegaram": 1871, "salvar": 6276, "soltaram": 158,
    "soltou": 402, "conhecer": 7428, "fumaça": 4, "geladeira": 922, "lâmpada": 7,
    "mulher": 37539, "papel": 6358, "perguntou": 3123, "sentiu": 2173, "sinal": 7249,
    "combinar": 358, "criança": 47, "estavam": 12098, "folhas": 1191, "pessoa": 19329,
    "convidar": 1103, "pegou": 7831, "descobrir": 7463, "passeando": 330, "passou": 8132,
    "próximo": 128, "aconteceu": 32406, "chegar": 19161, "ensinar": 2896, "ficaram": 2514,
    "seguir": 6428, "assustar": 1232, "começou": 28, "tarde": 27604, "limpar": 3284,
    "pequena": 9501, "pequeno": 12088, "assustado": 1754, "cantar": 4651, "policial": 6365,
    "subiu": 972, "feliz": 25486, "caçador": 1345, "sujo": 2178, "alguém": 1014,
    "esquece": 2681, "pegue": 12937, "quatro": 14016, "quero": 117209, "minha": 178027,
    "muitos": 15148, "nesse": 9927, "ninguém": 700, "pássaro": 38, "português": 17
}

# ============================================================================
# FUNCOES DE METRICAS
# ============================================================================

def estimar_silabas(palavra):
    palavra = palavra.lower()
    palavra = unicodedata.normalize('NFKD', palavra).encode('ASCII', 'ignore').decode('ASCII')
    
    vogais = 'aeiou'
    contagem = 0
    i = 0
    while i < len(palavra):
        if palavra[i] in vogais:
            contagem += 1
            if i + 1 < len(palavra) and palavra[i+1] in vogais:
                i += 1
        i += 1
    return max(1, contagem)

def estimar_derivacoes(palavra):
    palavra = palavra.lower()
    derivacoes = 0
    
    if palavra.endswith('ção') or palavra.endswith('são'):
        derivacoes += 3
    if palavra.endswith('mente'):
        derivacoes += 4
    if palavra.endswith('dade'):
        derivacoes += 3
    if palavra.endswith('vel'):
        derivacoes += 2
    if palavra.endswith('mento'):
        derivacoes += 3
    if palavra.endswith('ar') or palavra.endswith('er') or palavra.endswith('ir'):
        derivacoes += 2
    if palavra.endswith('ando') or palavra.endswith('endo') or palavra.endswith('indo'):
        derivacoes += 2
    if palavra.endswith('ado') or palavra.endswith('ido'):
        derivacoes += 2

    palavras_comuns = ['casa', 'homem', 'mulher', 'criança', 'escola', 'trabalho']
    if palavra in palavras_comuns:
        derivacoes += 3
    
    if len(palavra) <= 5:
        derivacoes += 2
    
    return min(10, max(0, derivacoes))

def contagem_encontros_complexos(palavra):
    palavra = palavra.lower()
    padroes_complexos = ['lh', 'nh', 'ch', 'rr', 'ss', 'tr', 'pr', 'br', 'cr', 'vr', 'fl', 'cl', 'pl', 'ps', 'sc', 'xc', 'pt']
    total = sum(1 for padrao in padroes_complexos if padrao in palavra)
    return total

def frequencia_de_uso(palavra):
    return frequencias.get(palavra.lower(), 0)

# ============================================================================
# PREPARAR DADOS PARA K-MEANS
# ============================================================================

print("Calculando metricas para todas as palavras...")

X_list = []
freq_max = max(frequencias.values()) if frequencias else 1
max_silabas = max(estimar_silabas(p) for p in palavras_teste)
max_derivacoes = max(estimar_derivacoes(p) for p in palavras_teste)

for palavra in palavras_teste:
    F = frequencia_de_uso(palavra) / freq_max
    silabas = estimar_silabas(palavra)
    CS = silabas / max_silabas
    derivacoes = estimar_derivacoes(palavra)
    SO = derivacoes / max_derivacoes if max_derivacoes > 0 else 0
    EC = contagem_encontros_complexos(palavra)
    
    if SO == 0:
        nivel = (0.30 * F) + (0.70 * CS)
    else:
        nivel = (0.20 * F) + (0.40 * CS) + (0.40 * SO)
    
    X_list.append([F, CS, SO, EC, nivel])

X = np.array(X_list)

X_kmeans = X[:, :4]
scaler = StandardScaler()
X_escalado = scaler.fit_transform(X_kmeans)

print(f"Metricas calculadas para {len(palavras_teste)} palavras")
print(f"Dimensoes da matriz: {X.shape}")
print("\nEstatisticas das metricas:")
print(f"  Frequencia - Min: {X[:,0].min():.4f}, Max: {X[:,0].max():.4f}, Media: {X[:,0].mean():.4f}")
print(f"  Silabas - Min: {X[:,1].min():.4f}, Max: {X[:,1].max():.4f}, Media: {X[:,1].mean():.4f}")
print(f"  Derivacoes - Min: {X[:,2].min():.4f}, Max: {X[:,2].max():.4f}, Media: {X[:,2].mean():.4f}")
print(f"  Encontros Complexos - Min: {X[:,3].min():.4f}, Max: {X[:,3].max():.4f}, Media: {X[:,3].mean():.4f}")
print(f"  Nivel de Dificuldade - Min: {X[:,4].min():.4f}, Max: {X[:,4].max():.4f}, Media: {X[:,4].mean():.4f}")

print("\nIniciando K-Means...")
time.sleep(1.0)

# ============================================================================
# CORES E CONFIGURACOES
# ============================================================================

cores_clusters = ['#FF6B6B', '#4ECDC4', '#FFD93D']

np.random.seed(42)
n_clusters = 3

indices_aleatorios = np.random.choice(len(X_escalado), n_clusters, replace=False)
centroides = X_escalado[indices_aleatorios].copy()

print("Centroides iniciais escolhidos aleatoriamente")
time.sleep(1.0)

labels_final = None
centroides_final = None

# ============================================================================
# FUNCAO PARA CRIAR GRAFICO INTERATIVO COM ZOOM FIXO
# ============================================================================

def criar_grafico_interativo(X, labels, centroides_reais, titulo, iteracao, pontos_mudaram=None):
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='black')
    ax.set_facecolor('black')
    
    cores_pontos = [cores_clusters[label] for label in labels]
    
    scatter = ax.scatter(X[:, 0], X[:, 3], 
                        c=cores_pontos,
                        s=100, 
                        alpha=0.7, 
                        edgecolors='white', 
                        linewidth=0.5,
                        picker=True)
    
    for j, centroide in enumerate(centroides_reais):
        tamanho_cluster = np.sum(labels == j)
        ax.scatter(centroide[0], centroide[3],
                  c=cores_clusters[j],
                  marker='X', 
                  s=400, 
                  edgecolors='white', 
                  linewidth=3,
                  label=f'Centroide {j+1} (n={tamanho_cluster})')
    
    if pontos_mudaram is not None and np.any(pontos_mudaram):
        pontos_mudaram_plot = X[pontos_mudaram]
        ax.scatter(pontos_mudaram_plot[:, 0], pontos_mudaram_plot[:, 3],
                  facecolors='none',
                  edgecolors='yellow',
                  s=180,
                  linewidth=2,
                  label=f'Pontos que mudaram ({np.sum(pontos_mudaram)})')
    
    # ========== LIMITES FIXOS DOS EIXOS ==========
    # X: 0 ate 0.3, Y: 0 ate 1.5
    ax.set_xlim(-0.02, 0.32)
    ax.set_ylim(-0.1, 1.6)
    
    ax.set_title(titulo, fontsize=16, fontweight='bold', color='white')
    ax.set_xlabel("Frequencia de Uso (Normalizada)", fontsize=12, color='white')
    ax.set_ylabel("Encontros Consonantais Complexos", fontsize=12, color='white')
    
    ax.tick_params(colors='white')
    
    legend = ax.legend(facecolor='black', edgecolor='white', loc='upper left')
    for text in legend.get_texts():
        text.set_color('white')
    
    ax.grid(True, linestyle='--', alpha=0.2, color='gray')
    
    fig.text(0.02, 0.02, "Use zoom (scroll) e arraste (pan) para explorar", 
             color='white', fontsize=10, alpha=0.7)
    
    fig.text(0.02, 0.06, "Zoom padrao: X[0,0.3] | Y[0,1.5]", 
             color='white', fontsize=9, alpha=0.5)
    
    def on_pick(event):
        ind = event.ind[0]
        palavra = palavras_teste[ind]
        freq = X[ind, 0]
        ec = X[ind, 3]
        nivel = X[ind, 4]
        ax.set_title(f"{palavra} - Freq: {freq:.3f}, EC: {ec:.0f}, Nivel: {nivel:.3f}", 
                     fontsize=14, fontweight='bold', color='white')
        fig.canvas.draw()
    
    fig.canvas.mpl_connect('pick_event', on_pick)
    
    plt.tight_layout()
    plt.show()
    plt.pause(0.1) 
    
    return fig, ax

# ============================================================================
# LOOP PRINCIPAL DO K-MEANS
# ============================================================================

for iteracao in range(1, 11):
    distancias = []
    for ponto in X_escalado:
        dist_ao_centroide = [np.linalg.norm(ponto - centroide) for centroide in centroides]
        distancias.append(dist_ao_centroide)
    
    distancias = np.array(distancias)
    labels = np.argmin(distancias, axis=1)

    novos_centroides = []
    for k in range(n_clusters):
        pontos_do_cluster = X_escalado[labels == k]
        if len(pontos_do_cluster) > 0:
            novo_centroide = np.mean(pontos_do_cluster, axis=0)
        else:
            novo_centroide = X_escalado[np.random.choice(len(X_escalado))]
        novos_centroides.append(novo_centroide)
    
    novos_centroides = np.array(novos_centroides)
    convergiu = np.allclose(centroides, novos_centroides, atol=1e-4)
    centroides = novos_centroides.copy()
    
    pontos_mudaram = None
    if iteracao > 1:
        pontos_mudaram = labels != labels_anterior
    
    centroides_reais = scaler.inverse_transform(centroides)
    titulo = f"K-Means - Iteracao {iteracao}"
    if convergiu:
        titulo += " - CONVERGIU"
    
    fig, ax = criar_grafico_interativo(X, labels, centroides_reais, titulo, iteracao, pontos_mudaram)
    
    print(f"\nIteracao {iteracao}:")
    for k in range(n_clusters):
        n_pontos = np.sum(labels == k)
        print(f"  Cluster {k+1}: {n_pontos} palavras")
    
    print("\nDistribuicao atual (primeiras 10 palavras por cluster):")
    for k in range(n_clusters):
        palavras_cluster = [palavras_teste[i] for i in range(len(palavras_teste)) if labels[i] == k]
        if len(palavras_cluster) > 0:
            mostrar = palavras_cluster[:10]
            if len(palavras_cluster) > 10:
                mostrar.append(f"... e mais {len(palavras_cluster)-10}")
            print(f"  Cluster {k+1}: {', '.join(mostrar)}")
    
    if iteracao == 10 or convergiu:
        labels_final = labels.copy()
        centroides_final = centroides.copy()
    
    if convergiu:
        print("\nAlgoritmo convergiu. Centroides estabilizados.")
        break
    
    labels_anterior = labels.copy()
    
    if iteracao < 10:
        print("\n" + "="*60)
        print("Proxima iteracao em 3 segundos...")
        time.sleep(3.0)
        clear_output(wait=True)

# ============================================================================
# RESULTADOS FINAIS
# ============================================================================

clear_output(wait=True)

print("="*60)
print("K-MEANS FINALIZADO")
print("="*60)

if labels_final is not None:
    plt.figure(figsize=(12, 7), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')
    
    cores_pontos_final = [cores_clusters[label] for label in labels_final]
    
    scatter = plt.scatter(X[:, 0], X[:, 3], 
                         c=cores_pontos_final,
                         s=130, 
                         alpha=0.8, 
                         edgecolors='white', 
                         linewidth=1)
    
    centroides_reais_final = scaler.inverse_transform(centroides_final)
    for j, centroide in enumerate(centroides_reais_final):
        tamanho_cluster = np.sum(labels_final == j)
        plt.scatter(centroide[0], centroide[3],
                   c=cores_clusters[j],
                   marker='X', 
                   s=500, 
                   edgecolors='white', 
                   linewidth=4,
                   label=f'Centroide {j+1} (n={tamanho_cluster})')
    
    # ========== LIMITES FIXOS DOS EIXOS ==========
    # X: 0 ate 0.3, Y: 0 ate 1.5
    ax.set_xlim(-0.02, 0.32)
    ax.set_ylim(-0.1, 1.6)
    
    plt.title("RESULTADO FINAL DO K-MEANS", fontsize=16, fontweight='bold', color='white')
    plt.xlabel("Frequencia de Uso (Normalizada)", fontsize=12, color='white')
    plt.ylabel("Encontros Consonantais Complexos", fontsize=12, color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')
    
    legend = plt.legend(facecolor='black', edgecolor='white', loc='upper left')
    for text in legend.get_texts():
        text.set_color('white')
    
    plt.grid(True, linestyle='--', alpha=0.2, color='gray')
    plt.tight_layout()
    plt.show()

# ============================================================================
# TABELA DE RESULTADOS
# ============================================================================

df_resultado = pd.DataFrame(palavras_teste, columns=['Palavra'])
df_resultado['Cluster'] = labels_final
df_resultado['Frequencia'] = X[:, 0]
df_resultado['Silabas'] = X[:, 1]
df_resultado['Derivacoes'] = X[:, 2]
df_resultado['Encontros_Complexos'] = X[:, 3].astype(int)
df_resultado['Nivel_Calculado'] = X[:, 4]

ranking_clusters = df_resultado.groupby('Cluster')['Encontros_Complexos'].mean().sort_values().index
mapeamento_niveis = {
    ranking_clusters[0]: "facil",
    ranking_clusters[1]: "medio",
    ranking_clusters[2]: "dificil"
}
df_resultado['Classificacao_KMeans'] = df_resultado['Cluster'].map(mapeamento_niveis)

def classificar_nivel(nivel):
    if nivel <= 0.3:
        return "facil"
    elif nivel <= 0.6:
        return "medio"
    else:
        return "dificil"

df_resultado['Classificacao_Calculada'] = df_resultado['Nivel_Calculado'].apply(classificar_nivel)

print("\nTABELA DE RESULTADOS:")
print("="*80)
print(f"Total de palavras analisadas: {len(df_resultado)}")

print("\nRESUMO POR CLUSTER (K-Means):")
print("="*80)
for nivel in ['facil', 'medio', 'dificil']:
    df_nivel = df_resultado[df_resultado['Classificacao_KMeans'] == nivel]
    palavras_nivel = df_nivel['Palavra'].tolist()
    print(f"\n{nivel.upper()} ({len(palavras_nivel)} palavras):")
    for i in range(0, len(palavras_nivel), 10):
        print(f"   {', '.join(palavras_nivel[i:i+10])}")

print("\nCOMPARACAO: K-Means vs Nivel Calculado")
print("="*80)
df_resultado['Match'] = df_resultado['Classificacao_KMeans'] == df_resultado['Classificacao_Calculada']
print(f"Match: {df_resultado['Match'].sum()}/{len(df_resultado)} ({df_resultado['Match'].mean()*100:.1f}%)")

print("\nPalavras com classificacao DIVERGENTE:")
divergentes = df_resultado[~df_resultado['Match']]
if len(divergentes) > 0:
    print(divergentes[['Palavra', 'Classificacao_Calculada', 'Classificacao_KMeans', 'Nivel_Calculado']].head(20).to_string(index=False))
    if len(divergentes) > 20:
        print(f"... e mais {len(divergentes)-20} palavras divergentes")
else:
    print("Todas as palavras tem classificacao consistente")

print("\n" + "="*80)
print(f"Analise concluida com {len(palavras_teste)} palavras")