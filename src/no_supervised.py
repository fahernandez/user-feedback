import pandas as pd
import plotly.express as px

results_clust = pd.read_csv('/data-analysis/include/clustering-metrics.csv')[[
        'homogeneity',
        'completeness',
        'v_measure',
        'vec_name',
        'fs_name',
        'clust_name',
        'repetition',
        'n_clusters',
        'noise']] \
        .assign(vec_tech=lambda x: x['vec_name'].apply(lambda y: y.split('-')[0])) \
        .assign(binary=lambda x: x['vec_name'].apply(lambda y: y.split('-')[1])) \
        .assign(k=lambda x: x['fs_name'].apply(lambda y: y.split('-')[1])) \
        .assign(
        clust_tech=lambda x: x['clust_name'].apply(lambda y: y.split('-')[0])) \
        [['homogeneity', 'completeness', 'v_measure', 'n_clusters',
          'noise', 'vec_tech', 'binary', 'k', 'clust_tech', 'clust_name']] \
        .rename(columns={
            'homogeneity': 'Homogeneidad',
            'completeness': 'Completitud',
            'v_measure': 'Medida-V',
            'n_clusters': 'Grupos',
            'vec_tech': 'Vectorización',
            'noise': 'Ruido',
            'binary': 'Binarización',
            'clust_tech': 'Agrupación',
            'clust_name': 'Detalle',
            'k': 'K'
        }) \
        .replace({
            'bow': 'BOW',
            'tf_idf': 'TF-IDF',
            'kmeans': 'K-Means',
            'dbscan': 'DBScan',
            'binary': 'Si',
            'non_binary': 'No',
            'no_binary': 'No',
            'all': 'Todos'
        })


def clust_model_comparisson_graph():
    clust_tech = pd.melt(results_clust.sort_values(by='Medida-V', ascending=False) \
                         .groupby('Agrupación') \
                         .first().reset_index() \
                             [['Agrupación', 'Homogeneidad', 'Completitud',
                               'Medida-V']],
                         id_vars=['Agrupación'],
                         value_vars=['Homogeneidad', 'Completitud', 'Medida-V'])

    no_sup_comp_fig = px.scatter(clust_tech.rename(columns={
        'variable': 'Índice',
        'value': 'Valor'
    }), x="Índice", y="Valor", color='Agrupación',
        title="Comparación de técnicas de agrupación")

    for g in no_sup_comp_fig.data:
        g.update(mode='markers')

    return no_sup_comp_fig


def k_model_vect():
    vect_tech = pd.melt(results_clust[results_clust['Agrupación'] == 'K-Means'] \
            .sort_values(by='Medida-V',ascending=False)\
            .groupby('Vectorización')\
            .first().reset_index()\
            [['Vectorización', 'Homogeneidad', 'Completitud', 'Medida-V']],
        id_vars=['Vectorización'],
        value_vars=['Homogeneidad', 'Completitud', 'Medida-V'])

    fig = px.scatter(vect_tech.rename(columns={
        'variable': 'Índice',
        'value': 'Valor'
    }), x="Índice", y="Valor", color='Vectorización',
        title="Comparación entre métodos de vectorización", text="Valor")
    for g in fig.data:
        g.update(mode='markers')

    return fig


def k_model_bin():
    binary_tech = pd.melt(results_clust[
        (results_clust['Agrupación'] == 'K-Means') &
        (results_clust['Vectorización'] == 'TF-IDF')
    ].sort_values(by='Medida-V',ascending=False) \
        .groupby('Binarización') \
        .first().reset_index() \
        [['Binarización', 'Homogeneidad', 'Completitud', 'Medida-V']],
    id_vars=['Binarización'],
    value_vars=['Homogeneidad', 'Completitud', 'Medida-V'])

    fig = px.scatter(binary_tech.rename(columns={
        'variable': 'Índice',
        'value': 'Valor'
    }), x="Índice", y="Valor", color='Binarización',
        title="Comparación de técnicas de binarización")

    for g in fig.data:
        g.update(mode='markers')

    return fig


def k_model_fec():
    k_fect = pd.melt(results_clust[
        (results_clust['Agrupación'] == 'K-Means') &
        (results_clust['Vectorización'] == 'TF-IDF') &
        (results_clust['Binarización'] == 'Si')]
        .sort_values(by='Medida-V',ascending=False)\
        .groupby('K')\
        .first().reset_index()\
    [['K', 'Homogeneidad', 'Completitud', 'Medida-V']],
    id_vars=['K'],
    value_vars=['Homogeneidad', 'Completitud', 'Medida-V'])

    fig = px.scatter(k_fect.rename(columns={
        'variable': 'Índice',
        'value': 'Valor'
    }), x="Índice", y="Valor", color='K',
        title="Comparación entre cantidad de features")

    for g in fig.data:
        g.update(mode='markers')

    return fig


def k_model_groups():
    clusters = pd.melt(results_clust[
        (results_clust['Agrupación'] == 'K-Means') &
        (results_clust['Vectorización'] == 'TF-IDF') &
        (results_clust['Binarización'] == 'Si') &
        (results_clust['K'] == '800')]\
            .sort_values(by='Medida-V',ascending=False)\
            .groupby('Grupos')\
            .first().reset_index()\
            [['Grupos', 'Homogeneidad', 'Completitud', 'Medida-V']],
        id_vars=['Grupos'],
        value_vars=['Homogeneidad', 'Completitud', 'Medida-V'])

    fig = px.line(clusters.rename(columns={
        'variable': 'Índice',
        'value': 'Valor'
    }), x="Índice", y="Valor", color='Grupos',
        title="Comparación entre cantidad de grupos")

    for g in fig.data:
        g.update(mode='markers')

    return fig


def db_model_vect():
    vect_tech = pd.melt(results_clust[results_clust['Agrupación'] == 'DBScan'] \
            .sort_values(by='Medida-V',ascending=False)\
            .groupby('Vectorización')\
            .first().reset_index()\
            [['Vectorización', 'Homogeneidad', 'Completitud', 'Medida-V']],
        id_vars=['Vectorización'],
        value_vars=['Homogeneidad', 'Completitud', 'Medida-V'])

    fig = px.scatter(vect_tech.rename(columns={
        'variable': 'Índice',
        'value': 'Valor'
    }), x="Índice", y="Valor", color='Vectorización',
        title="Comparación entre métodos de vectorización", text="Valor")
    for g in fig.data:
        g.update(mode='markers')

    return fig


def db_model_bin():
    binary_tech = pd.melt(results_clust[
        (results_clust['Agrupación'] == 'DBScan') &
        (results_clust['Vectorización'] == 'BOW')
    ].sort_values(by='Medida-V',ascending=False) \
        .groupby('Binarización') \
        .first().reset_index() \
        [['Binarización', 'Homogeneidad', 'Completitud', 'Medida-V']],
    id_vars=['Binarización'],
    value_vars=['Homogeneidad', 'Completitud', 'Medida-V'])

    fig = px.scatter(binary_tech.rename(columns={
        'variable': 'Índice',
        'value': 'Valor'
    }), x="Índice", y="Valor", color='Binarización',
        title="Comparación de técnicas de binarización")

    for g in fig.data:
        g.update(mode='markers')

    return fig


def db_model_fec():
    k_fect = pd.melt(results_clust[
        (results_clust['Agrupación'] == 'DBScan') &
        (results_clust['Vectorización'] == 'BOW') &
        (results_clust['Binarización'] == 'No')]
        .sort_values(by='Medida-V',ascending=False)\
        .groupby('K')\
        .first().reset_index()\
    [['K', 'Homogeneidad', 'Completitud', 'Medida-V']],
    id_vars=['K'],
    value_vars=['Homogeneidad', 'Completitud', 'Medida-V'])

    fig = px.scatter(k_fect.rename(columns={
        'variable': 'Índice',
        'value': 'Valor'
    }), x="Índice", y="Valor", color='K',
        title="Comparación entre cantidad de features")

    for g in fig.data:
        g.update(mode='markers')

    return fig


def db_model_noise():
    noise_fect = pd.melt(results_clust[
        (results_clust['Agrupación'] == 'DBScan') &
        (results_clust['Vectorización'] == 'BOW') &
        (results_clust['Binarización'] == 'No') &
        (results_clust['K'] == '800')
    ]
        .sort_values(by='Medida-V',ascending=False)\
        .groupby('Ruido')\
        .first().reset_index()\
    [['Ruido', 'Homogeneidad', 'Completitud', 'Medida-V']],
    id_vars=['Ruido'],
    value_vars=['Homogeneidad', 'Completitud', 'Medida-V'])

    fig = px.scatter(noise_fect.rename(columns={
        'variable': 'Índice',
        'value': 'Valor'
    }), x="Índice", y="Valor", color='Ruido',
        title="Comparación entre diferentes tolerancias de ruido")

    for g in fig.data:
        g.update(mode='markers')

    return fig

