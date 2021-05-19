import pandas as pd
import plotly.express as px


def clust_model_comparisson_graph():
    no_sup_comp = pd.read_csv('/data-analysis/include/clustering-metrics.csv')
    no_sup_comp = no_sup_comp[[
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
            'no_binary': 'No',
            'all': 'Todos'
        })

    clust_tech = pd.melt(no_sup_comp.sort_values(by='Medida-V', ascending=False) \
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
        title="Comparación de Técnicas de Agrupación")

    for g in no_sup_comp_fig.data:
        g.update(mode='markers')

    return no_sup_comp_fig
