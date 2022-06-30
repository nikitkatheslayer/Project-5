
from gensim.models import KeyedVectors

#загрузка модели
filename = "web_0_300_20.bin"
model = KeyedVectors.load_word2vec_format(filename, binary = True)
#поиск слова
result = model.most_similar(positive=['дом_NOUN'])
print(result)

#вычисление схожести для различных слов
#алгоритм вычисления схожести похожих слов

words = ['дом_NOUN', 'человек_NOUN', 'школа_NOUN', 'спорт_NOUN', 'окно_NOUN']
for word in words:
    if word in model:
        print(word)
        for i in model.most_similar(positive = [word], topn = 10):
            print(i[0], i[1])
        print ('\n')
    else:
        print(word + 'нет данного слова')


#визуализация с помощью tsne
keys = ['дом_NOUN', 'человек_NOUN', 'школа_NOUN', 'спорт_NOUN', 'окно_NOUN', 'дверь_NOUN', 'зуб_NOUN'
        , 'гитара_NOUN', 'сумка_NOUN', 'арбуз_NOUN', 'курица_NOUN']

#объединение в группы(чем больше данных, тем сложнее построить график. Близкие слова объединяются в группы  )
embedding_clusters = []
word_clusters = []
for word in keys:
    embeddings = []
    words = []
    for similar_word, _ in model.most_similar(word, topn=30):
        words.append(similar_word)
        embeddings.append(model[similar_word])
    embedding_clusters.append(embeddings)
    word_clusters.append(words)
    
from sklearn.manifold import TSNE
import numpy as np

#конфигурация tsne
#n_components — количество компонентов, т.е., размерность пространства значений;
#perplexity — перплексия, значение которой в t-SNE можно приравнять к эффективному количеству соседей.
#init — тип первоначальной инициализации векторов

embedding_clusters = np.array(embedding_clusters)
n, m, k = embedding_clusters.shape
tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

#построение плота
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def tsne_plot_similar_words(labels, embedding_clusters, word_clusters, a=0.7):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:,0]
        y = embeddings[:,1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2), 
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.grid(True)
    plt.title(r'Web. December 2014', fontsize=20)
    plt.savefig("plot.png", format='png', dpi=150, bbox_inches='tight')
    plt.show()

tsne_plot_similar_words(keys, embeddings_en_2d, word_clusters)



