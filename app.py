import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from src.transformation import fit_smv, predict
import plotly.express as px
import pandas as pd

# https://towardsdatascience.com/how-to-use-docker-to-deploy-a-dashboard-app-on-aws-8df5fb322708

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)
model = None

sup_comp = pd.read_csv('/code/include/model-comparison.csv')
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

no_sup_comp = pd.read_csv('/code/include/clustering-metrics.csv')
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

no_sup_comp_fig = px.line(clust_tech.rename(columns={
    'variable': 'Índice',
    'value': 'Valor'
}), x="Índice", y="Valor", color='Agrupación',
    title="Comparación de Técnicas de Agrupación")

for g in no_sup_comp_fig.data:
    g.update(mode='markers+lines')

app.layout = html.Div([
    dcc.Tabs(id='app-content-tabs', value='tab-1', children=[
        dcc.Tab(label='Predicción automática', value='tab-1'),
        dcc.Tab(label='Resultados Análisis Supervisado', value='tab-2'),
        dcc.Tab(label='Resultados Análisis No Supervisado', value='tab-3'),
    ], style={'padding-bottom': '10px'}),
    html.Div(id='app-content')
])


@app.callback(Output('app-content', 'children'),
              Input('app-content-tabs', 'value'))
def render_content(tab):
    if tab == 'tab-2':
        return html.Div([
            dcc.Graph(
                id='comparison-graph',
                figure=sup_comp_fig,
            )
        ], style={
            'margin': 'auto',
            'width': '50%',
            'border': "2px solid black",
            'padding': '10px'
        })
    elif tab == 'tab-3':
        return html.Div([
            dcc.Graph(
                id='no-sup-graph',
                figure=no_sup_comp_fig,
            )
        ], style={
            'margin': 'auto',
            'width': '50%',
            'border': "2px solid black",
            'padding': '10px'
        })
    elif tab == 'tab-1':
        return html.Div([
            html.H6("Predicción automática de comentarios!", style={
                "text-align": "center"
            }),
            html.Div(["Digite el comentario: ",
                      dcc.Input(
                          id='comment',
                          value='Mejorar letra de receta',
                          type='text')], style={
                "text-align": "center"
            }),
            html.Br(),
            html.Div([
                "Departamento de la Compañia: ", dcc.Dropdown(
                    id='creator-department',
                    options=[
                        {
                            'label': 'Customer Success',
                            'value': 'Customer Success'
                        },
                        {
                            'label': 'Sales',
                            'value': 'Sales'
                        },
                        {
                            'label': 'Other',
                            'value': 'Other'
                        }
                    ],
                    value='Customer Success'
                ),
                "Tipo de Recurso en CRM: ", dcc.Dropdown(
                    id='resource-type',
                    options=[
                        {
                            'label': 'Contact',
                            'value': 'contact'
                        },
                        {
                            'label': 'Deal',
                            'value': 'deal'
                        },
                        {
                            'label': 'Lead',
                            'value': 'lead'
                        }
                    ],
                    value='contact'
                )
            ], style={
                'margin': 'auto',
                'width': '50%',
                'border': "1px solid black",
                'padding': '10px'
            }),

            html.Div(id='true-label', style={
                'font-size': '2.5em',
                'text-align': 'center'
            }),
            html.Table([
                html.Tr([html.Td(['Agenda']), html.Td(id='agenda')]),
                html.Tr([html.Td(['Consulta']), html.Td(id='checkup')]),
                html.Tr([html.Td(['Facturación']), html.Td(id='billing')]),
                html.Tr([html.Td(['Información Médica']),
                         html.Td(id='medical-record')]),
                html.Tr([html.Td(['Perfil']), html.Td(id='profile')]),
                html.Tr([html.Td(['Servicio']), html.Td(id='service')]),
            ], style={
                "margin": "auto",
                'width': "70%"
            })
        ], style={
            'margin': 'auto',
            'width': '50%',
            'border': "2px solid black",
            'padding': '10px'
        }
        )


@app.callback(
    Output(component_id='true-label', component_property='children'),
    Output(component_id='agenda', component_property='children'),
    Output(component_id='checkup', component_property='children'),
    Output(component_id='billing', component_property='children'),
    Output(component_id='medical-record', component_property='children'),
    Output(component_id='profile', component_property='children'),
    Output(component_id='service', component_property='children'),
    Input(component_id='comment', component_property='value'),
    Input(component_id='creator-department', component_property='value'),
    Input(component_id='resource-type', component_property='value'),
)
def update_output_div(comment, creator_department, resource_type):
    cat, prob = predict(model, comment, creator_department, resource_type)
    return cat, prob[0], prob[1], prob[2], prob[3], prob[4], prob[5]


if __name__ == '__main__':
    model = fit_smv()
    app.run_server(host='0.0.0.0', port=5050, debug=True)
