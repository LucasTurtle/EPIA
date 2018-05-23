# Bibliotecas de estruturas de dados e manipulação algébrica
import numpy as np
import pandas as pd
import random
import math

# bibliotecas de projeção
import matplotlib.pyplot as plt # plot
import matplotlib.cm as cm 		# colour map

import plotly.offline as py

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

import time
import sys

np.set_printoptions(precision=4)

# Carrego uma tabela num dataframe
def loadData(path):
	df = pd.read_csv(path, sep='\t', dtype = float, index_col=False)
	return df

# Função que retorna o tempo desde o inicio da execução em X minutos : Y segundos 
def getElapsedTime():
	end = time.time()
	temp = end-start

	hours = temp//3600
	temp = temp - 3600*hours
	minutes = temp//60
	seconds = temp - 60*minutes
	total = ('%d min e %d seg' %(minutes,seconds))
	return total

# Retorna o valor mais alto de um dataframe
def highestValue(df):
	seriesHighest = df.max(numeric_only=True)
	return seriesHighest.max()

def euclideanDistance(a, b):
	result = 0
	for i in range (0, len(a)):
		result += ((a[i] - b[i]) ** 2)
	return (result ** (1/2))

def cosineSimilarity(a, b):
	prodEscalar = np.inner(a, b)
	(magnitudeA, magnitudeB) = 0, 0
	for i in range (0, len(a)):
		magnitudeA += a[i] ** 2
		magnitudeB += b[i] ** 2
	magnitudeA = magnitudeA ** (1/2)
	magnitudeB = magnitudeB ** (1/2)
	return (prodEscalar/(magnitudeA * magnitudeB))

# Acha a distancia entre um ponto e o seu centroide mais próximo
def distanceFromClosestCentroid(centroids, point):
	min_dist = float("inf")
	for centroid in centroids:
		# Se ainda não houver sido calculado o centroide (todos os valores dele forem 0), passe para a próxima iteração
		if (~centroid.any(axis=0)).any():
			continue
		dist = euclideanDistance(centroid, point)
		min_dist = dist if dist < min_dist else min_dist
	return min_dist

# Fitness Proportionate Selection
def rouletteWheel(distances, totalPoints):
	cumulativeProb = distances/np.sum(distances)
	p = np.random.random_sample()
	sum = 0
	for i in range (0, totalPoints):
		if p <= sum:
			return i
		sum += cumulativeProb[i]

# inicialização kmeans++
def smartCentroids(max_clusters, df):
	print ("smartCentroids")
	totalPoints = len(df)
	randomPoint = np.random.randint(0, totalPoints) # indice do ponto aleatório

	smartCentroids = np.zeros([max_clusters, len(df.columns)], dtype = float)
	smartCentroids[0] = df.iloc[randomPoint] # centroide designado no ponto aleatório

	# distancia quadrada entre o ponto e o seu centroide mais próximo 
	distances = np.full(totalPoints, -1, dtype = float)
	for i in range(1, max_clusters):
		for j in range (0, totalPoints):
			point = df.values[j]
			distances[j] = distanceFromClosestCentroid(smartCentroids, point) ** 2
		
		index = rouletteWheel(distances, totalPoints) # indice do novo centroide
		smartCentroids[i] = df.iloc[index]
	return smartCentroids

# Retorna uma lista em que cada linha é um centroide inicializado com valores random
def randomCentroids(max_clusters, df):
	print ("randomCentroids")
	maxValue = highestValue(df)
	columns = len(df.columns)
	randCentroids = np.zeros([max_clusters, columns], dtype = float)
	for i in range(0, max_clusters):
		for j in range(0, columns):
			randCentroids[i][j] = round(random.uniform(0, maxValue), 4) # coordenada de valor float aleatorio com 4 casas decimais de precisão
	return randCentroids

# retorna qual a distancia de cada elemento para cada cluster e quantos elementos existem em cada cluster
def calculateClusters(df, centroids, max_clusters, mode):
	clusters = np.full((df.shape[0], max_clusters), -1).astype(float)
	elements_in_cluster = np.zeros(shape=[max_clusters])
	doc_in_cluster = np.zeros((df.shape[0], 2), dtype = float) # em que cluster está tal doc, e qual a distancia do doc ao cluster

	for i in range(df.shape[0]):       
		doc = df.values[i]     # coordenada de um doc    
		centroid_index = 0
		min_dist = float("inf")

		# acha o centroide mais próximo de um ponto
		for centroid in centroids:
			# checa o modo e utiliza a distancia requisitada
			dist = cosineSimilarity(centroid, doc) if mode == "Cosine" else euclideanDistance(centroid, doc)

			if (dist < min_dist):
				min_dist = dist
				min_dist_index = centroid_index
			centroid_index += 1

		clusters[i][min_dist_index] = min_dist
		elements_in_cluster[min_dist_index] += 1
		doc_in_cluster[i][0] = min_dist_index
		doc_in_cluster[i][1] = min_dist
	return (clusters, elements_in_cluster, doc_in_cluster)

# recalcula a nova posição dos centroides, os movendo para o centro de seu grupo
def recalculateCentroids(df, max_clusters, centroids, clusters, elements_in_cluster):
	total_docs = clusters.shape[0]
	total_words = df.shape[1]
	new_centroids = np.zeros([max_clusters, total_words], dtype = float)

	# move o centroide para a posição média em seu cluster
	for i in range(0, total_docs):
		for j in range(0, max_clusters):
			if clusters[i][j] != -1:
				for k in range(0, total_words):
					new_centroids[j][k] += df.iloc[i][k] / elements_in_cluster[j]

			# se um cluster não tiver nenhum elemento, não movemos o centroide
			if (elements_in_cluster[j] == 0):
				new_centroids[j] = centroids[j]
	return new_centroids

# dada a matriz de cluters e posições, retorna somente o cluster desejado
def getCluster(allClusters, cluster_index, df):
	allClusters = pd.DataFrame(allClusters)
	cluster = allClusters.loc[allClusters[cluster_index] != -1]
	
	header_list = list(range(0, len(df.columns)))
	result = pd.DataFrame(columns=header_list)
	#print (cluster)

	for index, row in cluster.iterrows():
		result.loc[index] = df.loc[index].values
	return result


# calcula o indice silhouette
def silhouette (clusters, centroids, df, doc_in_cluster, max_clusters):
	total_elements = len(clusters)
	total_clusters = len(centroids)
	silhouette = np.zeros((total_elements, 2))

	for i in range(0, max_clusters):
		cluster = getCluster(clusters, i, df).round(4)
		for index, value in cluster.iterrows():
			a, b = (0, 0)

			# calcula a distancia média de um elemento pros outros elementos
			for jindex, value in cluster.iterrows():
				a += euclideanDistance (cluster.loc[index].values, cluster.loc[jindex].values)
			
			# se o cluster só tiver um elemento, eu calculo sua distancia com o centroide do seu grupo 
			if (len(cluster) == 1):
				a = euclideanDistance (cluster.loc[index].values, centroids[i])
			else:
				a = a/(len(cluster)-1)

			min_b = float("inf")
			# calcula a distancia minima de um elemento pros elementos de outros clusters
			for j in range (0, total_clusters):
				if (j == i): continue
				otherCluster = getCluster(clusters, j, df).round(4)

				# Se o outro cluster for vazio, eu calculo a distancia dos elementos deste cluster com os outros centroides
				if not otherCluster.empty:		
					for jindex, value in otherCluster.iterrows():
						b += euclideanDistance (cluster.loc[index].values, otherCluster.loc[jindex].values)
				else:
					b += euclideanDistance (cluster.loc[index].values, centroids[j])
			b = b/len(otherCluster)
			if (b < min_b): min_b = b
			#print (index)

			silhouette[index][0] = (min_b-a)/max(a, min_b)
			silhouette[index][1] = i
	silhouette = pd.DataFrame(silhouette)
	silhouette = silhouette.rename(silhouette.iloc[:, 1])
	silhouette = silhouette.rename(columns={0: 'Silhouette', 1: 'Clusters'})
	return silhouette

# Acha a magnitude dos centroides e retorna a variação de distancia entre eles
def errorFunction(old_centroids, new_centroids):
	magnitudeA, magnitudeB = (0, 0)
	for i in range (0, len(old_centroids)):
		magnitudeA += old_centroids[i] ** 2
		magnitudeB += new_centroids[i] ** 2
	magnitudeA = magnitudeA ** (1/2)
	magnitudeB = magnitudeB ** (1/2)
	return euclideanDistance(magnitudeA, magnitudeB)

# Reduz as dimensões do conjunto de dados para podermos visualizá-lo após a execução do algoritmo
def reduceDimensions(data, dims):
	X_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(data)
	X_embedded = TSNE(n_components=dims, perplexity=40, verbose=2).fit_transform(X_reduced)
	X_embedded = pd.DataFrame(X_embedded)
	df = X_embedded.rename(index=str, columns={0: "X", 1: "Y"})
	df.index = df.index.map(int)
	return df

def visualize(sil, df, centroids, max_clusters, iters, elapsedTime, distance_mode, centroids_mode, save_to):
	avg_sil = np.average(sil['Silhouette'])

	# Cria um subplot com 1 linha e dois gráficos
	fig, (ax1, ax2) = plt.subplots(1, 2)
	fig.set_size_inches(18, 7)

	# Plota o ìndice Silhouette com alturas entre -0.2 e 1
	ax1.set_xlim([-0.2, 1])

	# Insere um espaço em branco entre os clusters pra ficar clara a demarcação deles
	ax1.set_ylim([0, len(sil) + (max_clusters + 1) * 10])

	plt.suptitle(("Análise Silhouette do K-Means "
				  "com %d clusters" % max_clusters),
				 fontsize=14, fontweight='bold')

	# Loop para inserir a barra do silhouette de cada elemento
	y_lower = 10
	for cluster in sil.Clusters.unique():
		cluster_silhouette_values = sil.Silhouette.loc[cluster].values
		cluster_silhouette_values.sort() # Ordena os valor de cada elemento no cluster

		size_cluster = len(cluster_silhouette_values)
		y_upper = y_lower + size_cluster

		color = cm.spectral(float(cluster) / max_clusters)
		ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

		# Label the silhouette plots with their cluster numbers at the middle
		ax1.text(-0.05, y_lower + 0.5 * size_cluster, str(int(cluster)))

		# Compute the new y_lower for next plot
		y_lower = y_upper + 10  # 10 for the 0 samples

	ax1.set_title("Indice Silhouette de Diversos Clusters.")
	ax1.set_xlabel("Coeficientes Silhouette")
	ax1.set_ylabel("Indice do Cluster")

	# Linha que demarca a média dos valores silhouette
	ax1.axvline(x=avg_sil, color="red", linestyle="--")

	ax1.set_yticks([]) # Limpa o Eixo Y do gráfico 1
	ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]) # Demarca o eixo X do gráfico 1
	

	# Segundo gráfico mostrando a plotagem dos clusters
	colors = cm.spectral(sil.Clusters.values / max_clusters)
	ax2.scatter(df.X.values, df.Y.values, marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

	ax2.scatter(centroids[:, 0], centroids[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')

	for i, c in enumerate(centroids):
		ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

	ax2.set_title("K-Means com %s e Inicialização de Centroides %s.\n Convergiu em %d iterações com tempo de %s" % (distance_mode, centroids_mode, iters, elapsedTime))
	fig.savefig(save_to)
	plt.show()
	print (save_to)


def kmeans(df, max_clusters, max_iterations, threshold, centroids_mode, distance_mode, save_to):
	centroids = randomCentroids(max_clusters, df) if centroids_mode == "Random" else smartCentroids(max_clusters, df)

	iters = 0
	while iters < max_iterations:
		(clusters, elements_in_cluster, doc_in_cluster) = calculateClusters(df, centroids, max_clusters, distance_mode)

		new_centroids = recalculateCentroids(df, max_clusters, centroids, clusters, elements_in_cluster)
		error = errorFunction(centroids, new_centroids)
		
		print ("elementos em cada grupo: %s\terro: %s" % (elements_in_cluster, error))

		if (error <= threshold):
			break
		centroids = new_centroids
		iters += 1

	sil = silhouette(clusters, centroids, df, doc_in_cluster, max_clusters)
	a = df.assign(Clusters=doc_in_cluster[:, 0])
	elapsedTime = getElapsedTime()

	distance_mode = "Similaridade Cosseno" if distance_mode == "Cosine" else "Distância Euclidiana"
	centroids_mode = "Aleatórios" if centroids_mode == "Random" else "K-Means++"
	visualize(sil, df, centroids, max_clusters, iters, elapsedTime, distance_mode, centroids_mode, save_to)

def setProcessType(format_type):
	if format_type == 'bin':
		return "binary.tsv"
	elif format_type == 'tf':
		return "tf.tsv"
	else:
		return "tfidf.tsv"

def argError():
	print ("O programa precisa de 4 argumentos\npython kmeans.py\t1\t2\t3\t4\n")
	print ("1 - Tamanho da corpora pre-processada: Maior(big) ou  Menor(small). Default: small\n2 - Tipo do pre-processamento: Binário(bin), TF(tf) ou TF-IDF(tfidf). Default: tfidf\n3 - Metrica de Distancia: Euclidiana(euclidean) ou Cosseno(cos). Default: euclidean\n4 - Inicialização dos Centroides: Aleatorios(random) ou K-Means++(smart). Default: smart\n\nPor Exemplo:\npython kmeans.py big tfidf euclidean smart")
	exit()

def main(argv):
	if len(argv) < 5:
		argError()

	path = '../pre_process/processedBig/' if argv[1].lower() == "big" else '../pre_process/processedSmall/'
	format_type = setProcessType(argv[2].lower())
	path = path + format_type

	#df = pd.read_csv(path, sep='\t', index_col=0)
	df = pd.DataFrame(np.random.uniform(0, 500, size = (100, 55)))

	max_clusters = 2
	max_iterations = 100
	threshold = 0.1
	distance_mode = "Cosine" if argv[3].lower() == "cosine" else "Euclidean"
	centroids_mode = "Random" if argv[4].lower() == "random" else "Smart"

	save_to = '/big_results/' if argv[1].lower() == "big" else 'small_results/'
	file_name = argv[1].title() + argv[2].title() + distance_mode + centroids_mode + '.png'
	save_to += file_name


	df = reduceDimensions(df, 2)
	global start
	start = time.time()
	doc_in_cluster = kmeans(df, max_clusters, max_iterations, threshold, centroids_mode, distance_mode, save_to)

if __name__ == '__main__':
	main(sys.argv)