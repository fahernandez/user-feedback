import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from src.transformation import fit_smv, predict
from src.no_supervised import clust_model_comparisson_graph
from src.supervised import class_model_comparisson
from src.descriptive import get_desc_graph


# https://towardsdatascience.com/how-to-use-docker-to-deploy-a-dashboard-app-on-aws-8df5fb322708

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)
model = None

app.layout = html.Div([
    dcc.Tabs(id='app-content-tabs', value='tab-1', children=[
        dcc.Tab(label='Predicción automática', value='tab-1'),
        dcc.Tab(label='Resultados análisis supervisado', value='tab-2'),
        dcc.Tab(label='Resultados análisis no supervisado', value='tab-3'),
        dcc.Tab(label='Análisis descriptivo', value='tab-4'),
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
                figure=class_model_comparisson(),
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
                figure=clust_model_comparisson_graph(),
            )
        ], style={
            'margin': 'auto',
            'width': '50%',
            'border': "2px solid black",
            'padding': '10px'
        })
    elif tab == 'tab-4':
        return html.Div(
            [
                html.Div([
                    "Departamento de la Compañia: ", dcc.Dropdown(
                        id='creator-department-desc',
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
                            },
                            {
                                'label': 'All',
                                'value': 'all'
                            }
                        ],
                        value='all'
                    ),
                    "Tipo de Recurso en CRM: ", dcc.Dropdown(
                        id='resource-type-desc',
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
                            },
                            {
                                'label': 'All',
                                'value': 'all'
                            }
                        ],
                        value='all'
                    ),
                    "Parte del producto: ", dcc.Dropdown(
                        id='product-part-desc',
                        options=[
                            {
                                'label': 'Agenda',
                                'value': 'agenda'
                            },
                            {
                                'label': 'Consulta',
                                'value': 'consulta'
                            },
                            {
                                'label': 'Facturación',
                                'value': 'facturación'
                            },
                            {
                                'label': 'Información médica',
                                'value': 'información médica'
                            },
                            {
                                'label': 'Perfil',
                                'value': 'perfil'
                            },
                            {
                                'label': 'Servicio',
                                'value': 'servicio'
                            },
                            {
                                'label': 'All',
                                'value': 'all'
                            }
                        ],
                        value='all'
                    )
                ]),
                html.Div([
                    dcc.Graph(
                        id='des-uni-freq-graph'
                    ),
                    dcc.Graph(
                        id='des-biuni-freq-graph'
                    ),
                    dcc.Graph(
                        id='des-uni-cloud-graph'
                    ),
                    dcc.Graph(
                        id='des-biuni-cloud-graph'
                    )
                ])
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
                    id='creator-department-pred',
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
                    id='resource-type-pred',
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
    Input(component_id='creator-department-pred', component_property='value'),
    Input(component_id='resource-type-pred', component_property='value'),
)
def update_output_div(comment, creator_department, resource_type):
    cat, prob = predict(model, comment, creator_department, resource_type)
    return cat, prob[0], prob[1], prob[2], prob[3], prob[4], prob[5]


@app.callback(
    Output(component_id='des-uni-freq-graph', component_property='figure'),
    Output(component_id='des-biuni-freq-graph', component_property='figure'),
    Output(component_id='des-uni-cloud-graph', component_property='figure'),
    Output(component_id='des-biuni-cloud-graph', component_property='figure'),
    Input(component_id='product-part-desc', component_property='value'),
    Input(component_id='creator-department-desc', component_property='value'),
    Input(component_id='resource-type-desc', component_property='value'),
)
def update_descriptive(part, creator, resource):
    fig1, fig2, fig3, fig4 = get_desc_graph(part, creator, resource)
    return fig1, fig2, fig3, fig4


if __name__ == '__main__':
    model = fit_smv()
    app.run_server(host='0.0.0.0', port=5050, debug=True)
