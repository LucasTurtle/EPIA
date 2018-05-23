Para rodar o programa, precisamos de 4 argumentos, no formato abaixo:
python kmeans.py  1 2 3 4

Os argumentos são:
1 - Tamanho da corpora pre-processada: Maior(big) ou  Menor(small). Default: small
2 - Tipo do pre-processamento: Binário(bin), TF(tf) ou TF-IDF(tfidf). Default: tfidf
3 - Metrica de Distancia: Euclidiana(euclidean) ou Cosseno(cos). Default: euclidean
4 - Inicialização dos Centroides: Aleatorios(random) ou K-Means++(smart). Default: smart

Por Exemplo:
python kmeans.py big tfidf euclidean smart

Se a corpora a ser processada for a Maior(big), o pós-processamento será salvo no diretório big_results. Caso ela seja a Menor(small), ele será salvo no diretório small_results.