import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
from plotly.subplots import make_subplots


def class_model_comparisson():
    sup_comp = pd.read_csv('/data-analysis/include/model-comparison.csv')
    sup_comp = sup_comp[['mean_test', 'std_test', 'repetition', 'model']]
    sup_comp_fig = px.line(sup_comp.rename(columns={
        'mean_test': "Precisión Media",
        "repetition": "Repeticiones",
        "model": "Modelo"
    }),
        x="Repeticiones",
        y="Precisión Media",
        color="Modelo",
        title="Comparación entre modelos")
    for g in sup_comp_fig.data:
        g.update(mode='markers+lines')

    return sup_comp_fig


def best_vec_model_graph(model_result):
    best_bow = model_result[model_result['combination'] == 'bow'].sort_values(by=['repetition','rank_test_score'], ascending=True).groupby('repetition').first().reset_index()
    best_tfidf = model_result[model_result['combination'] == 'tf-idf'].sort_values(by=['repetition','rank_test_score'], ascending=True).groupby('repetition').first().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=best_bow['repetition'], y=best_bow['mean_test_score'],
                        mode='lines+markers+text',
                        name='Bag of Words',
                        text=best_bow['mean_test_score'].apply(lambda w: math.floor(w * 10 ** 3) / 10 ** 3).to_frame(),
                        textposition="bottom center"))
    fig.add_trace(go.Scatter(x=best_tfidf['repetition'], y=best_tfidf['mean_test_score'],
                        mode='lines+markers+text',
                        name='TF-IDF',
                        text=best_tfidf['mean_test_score'].apply(lambda w: math.floor(w * 10 ** 3) / 10 ** 3).to_frame(),
                        textposition="bottom center"))
    # Edit the layout
    fig.update_layout(title='',
                       xaxis_title='Repeticiones',
                       yaxis_title='Precisión media en grupo de validación')

    return fig


def tunning_graph(result, combination, param, name):
    result["mean_test_score"] = result["mean_test_score"].apply(lambda w: math.floor(w * 10 ** 3) / 10 ** 3)

    fig = px.line(result[result['combination'] == combination].sort_values(by=['repetition', param,'rank_test_score'], ascending=True).groupby(['repetition', param]).first().reset_index().rename(columns={'repetition':"Repeticiones", param:name, "mean_test_score":"Precisión media en grupo de validación"}), x="Repeticiones", y="Precisión media en grupo de validación".format(name), color=name, title="",text="Precisión media en grupo de validación")
    fig.update_traces(textposition='top center')
    return fig


def get_text(score, cut):
    res = score.apply(lambda w: math.floor(w * 10 ** 3) / 10 ** 3).to_frame()
    row, col = res.shape
    for i in range(0,row):
        if res.iloc[i,0] <= cut:
            res.iloc[i,:] = ''
        if i % 3 != 0:
            res.iloc[i,:] = ''
    return res


def complexity_graph(features, result, combination, cut):
    to_render = len(features.items())
    fig = make_subplots(rows=int(to_render+1/2 if to_render%2==1 else to_render/2), cols=2)
    count = 1
    row_count = 1
    col_count = 1
    for f, label in features.items():
        res = result[result['combination']==combination]\
            .sort_values(by='rank_test_score')\
            .groupby([f])\
            .first()\
            .reset_index()

        fig.add_trace(go.Scatter(x=res[f], y=res['mean_train_score'],
                    mode='lines+text',
                    name='Entrenamiento',
                    text=get_text(res['mean_train_score'], cut),
                    textposition="top center",
                    showlegend=count==1,
                    line=dict(color='royalblue', width=2)), row=row_count, col=col_count)

        fig.add_trace(go.Scatter(x=res[f], y=res['mean_test_score'],
                    mode='lines+text',
                    name='Validación',
                    text=get_text(res['mean_test_score'], cut),
                    textposition="top center",
                    showlegend=count==1,
                    line=dict(color='firebrick', width=2)), row=row_count, col=col_count)

        # Update xaxis properties
        fig.update_xaxes(title_text=label, row=row_count, col=col_count)

        # Update yaxis properties
        fig.update_yaxes(title_text="Precisión", row=row_count, col=col_count)

        count+=1
        row_count = row_count + 1 if count%2==1 else row_count
        col_count = col_count + 1 if count%2==0 else col_count

    # Edit the layout
    fig.update_layout(title='Complejidad del modelo')
    return fig