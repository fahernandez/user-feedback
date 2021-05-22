import numpy as np
import plotly.express as px
import plotly
import plotly.graph_objs as go
import random
from nltk import word_tokenize, bigrams, FreqDist
from nltk.util import ngrams


def get_desc_graph(part, creator, resource, comments):
    unigramdist = FreqDist()
    bigramfdist = FreqDist()
    fc = comments

    if part != 'all':
        fc = fc[fc['primary_category'] == part]

    if creator != 'all':
        fc = fc[fc['creator_department'] == creator]

    if resource != 'all':
        fc = fc[fc['resource_type'] == resource]

    for index, sentence in fc.iterrows():
        unigrams = ngrams(sentence['tokens'], 1)
        bigrams = ngrams(sentence['tokens'], 2)
        unigramdist.update(unigrams)
        bigramfdist.update(bigrams)

    return unigram_freq_graph(unigramdist), bigram_freq_graph(bigramfdist), \
           unigram_word_cloud(unigramdist), bigram_word_cloud(bigramfdist)


def unigram_freq_graph(unigramdist):
    mostcommon = unigramdist.most_common(10)
    x0 = np.array([i[0] for i, v in mostcommon])
    x1 = np.array([v for i, v in mostcommon])
    fig = px.histogram(x=x0, y=x1)
    fig.update_layout(
        title_text='Frecuencia de unigramas',  # title of plot
        xaxis_title_text='Palabra',  # xaxis label
        yaxis_title_text='Frecuencia',  # yaxis label
        bargap=0.05,  # gap between bars of adjacent location coordinates
        bargroupgap=0.05  # gap between bars of the same location coordinates
    )

    return fig


def bigram_freq_graph(bigramfdist):
    mostcommon = bigramfdist.most_common(10)
    x0 = np.array([i[0] + ' ' + i[1] for i, v in mostcommon])
    x1 = np.array([v for i, v in mostcommon])
    fig = px.histogram(x=x0, y=x1)
    fig.update_layout(
        title_text='Frecuencia de bigramas',  # title of plot
        xaxis_title_text='Palabra',  # xaxis label
        yaxis_title_text='Frecuencia',  # yaxis label
        bargap=0.05,  # gap between bars of adjacent location coordinates
        bargroupgap=0.05  # gap between bars of the same location coordinates
    )

    return fig


def unigram_word_cloud(unigramdist):
    filter_words = dict([(m, n) for (m,), n in unigramdist.items() if n > 20])

    words = list(filter_words.keys())
    colors = [plotly.colors.DEFAULT_PLOTLY_COLORS[random.randrange(1, 10)] for i
              in range(len(filter_words))]
    weights = [x / 5 for x in list(filter_words.values())]

    data = go.Scatter(x=[random.random() for i in range(len(filter_words))],
                      y=[random.random() for i in range(len(filter_words))],
                      mode='text',
                      text=words,
                      marker={'opacity': 0.3},
                      textfont={'size': weights,
                                'color': colors})
    layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False,
                                  'zeroline': False},
                        'yaxis': {'showgrid': False, 'showticklabels': False,
                                  'zeroline': False}})
    fig = go.Figure(data=[data], layout=layout)

    return fig


def bigram_word_cloud(bigramfdist) -> object:
    filter_words = dict([(m + ' ' + y, n) for (m,y), n in bigramfdist.items() if n > 10])

    words = list(filter_words.keys())
    colors = [plotly.colors.DEFAULT_PLOTLY_COLORS[random.randrange(1, 10)] for i
              in range(len(filter_words))]
    weights = list(filter_words.values())

    data = go.Scatter(x=[random.random() for i in range(30)],
                      y=[random.random() for i in range(30)],
                      mode='text',
                      text=words,
                      marker={'opacity': 0.3},
                      textfont={'size': weights,
                                'color': colors})
    layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False,
                                  'zeroline': False},
                        'yaxis': {'showgrid': False, 'showticklabels': False,
                                  'zeroline': False}})
    fig = go.Figure(data=[data], layout=layout)

    return fig
