import pandas as pd
import plotly.express as px


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
        title="Comparación entre Modelos")
    for g in sup_comp_fig.data:
        g.update(mode='markers+lines')

    return sup_comp_fig
