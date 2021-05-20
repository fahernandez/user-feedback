import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
from src.transformation import fit_smv, predict
from src.no_supervised import clust_model_comparisson_graph
from src.supervised import class_model_comparisson
from src.descriptive import get_desc_graph

# https://towardsdatascience.com/how-to-use-docker-to-deploy-a-dashboard-app-on-aws-8df5fb322708

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.LUMEN],
                # external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)
model = None

app.layout = html.Div([
    dbc.Tabs(id="tabs", active_tab="pred", children=[
            dbc.Tab(label="Predicción automática", tab_id="pred"),
            dbc.Tab(label="Resultados análisis supervisado", tab_id="sup"),
            dbc.Tab(label="Resultados análisis no supervisado", tab_id="no-sup"),
            dbc.Tab(label="Análisis descriptivo", tab_id="desc"),
        ]),
    dbc.Card(id="content", body=True)
])


@app.callback(Output('content', 'children'),
              Input('tabs', 'active_tab'))
def render_content(tab):
    if tab == 'sup':
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
    elif tab == 'no-sup':
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
    elif tab == 'desc':
        return html.Div(
            [
                html.Div([
                    html.Div([
                        html.P("Departamento de la Compañia: ", style={
                            "margin-top": "0.5rem",
                            "margin-bottom": "0.5rem"
                        }),
                        dcc.Dropdown(
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
                    ], style={
                        'display': 'inline-block',
                        'width': '30%',
                        'margin': '0 10px'
                    }),
                    html.Div([
                        html.P("Tipo de Recurso en CRM: ", style={
                            "margin-top": "0.5rem",
                            "margin-bottom": "0.5rem"
                        }),
                        dcc.Dropdown(
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
                    ], style={
                        'display': 'inline-block',
                        'width': '30%',
                        'margin': '0 10px'
                    }),
                    html.Div([
                        html.P("Parte del producto: ", style={
                            "margin-top": "0.5rem",
                            "margin-bottom": "0.5rem"
                        }),
                        dcc.Dropdown(
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
                    ], style={
                        'display': 'inline-block',
                        'width': '30%',
                        'margin': '0 10px'
                    })
                ], style={
                    'border': "1px solid black",
                    'padding': '10px'
                }),
                html.Div([
                    html.Div([
                        html.Div([
                            dcc.Graph(
                                id='des-uni-freq-graph'
                            ),
                        ], style={
                            'display': 'inline-block',
                            'width': '49%'
                        }),
                        html.Div([
                            dcc.Graph(
                                id='des-biuni-freq-graph'
                            )
                        ], style={
                            'display': 'inline-block',
                            'width': '49%'
                        }),
                    ]),
                    html.Div([
                        html.Div([
                            dcc.Graph(
                                id='des-uni-cloud-graph'
                            ),
                        ], style={
                            'display': 'inline-block',
                            'width': '49%'
                        }),
                        html.Div([
                            dcc.Graph(
                                id='des-biuni-cloud-graph'
                            )
                        ], style={
                            'display': 'inline-block',
                            'width': '49%'
                        }),
                    ]),
                ])
            ], style={
                'margin': 'auto',
                'width': '100%',
            })
    elif tab == 'pred':
        return dbc.Card([
            html.P("Predicción automática de comentarios!", style={
                "text-align": "center",

            }),
            html.Div([
                html.P("Digite el comentario: ", style={
                    "display": "inline-block"
                }),
                dbc.Input(
                    id='comment',
                    value='Mejorar letra de receta',
                    type='text', style={
                        "display": "inline-block",
                        "width": "60%",
                        "margin": "0px 15px",
                    })],
                style={
                    "text-align": "center",
                }
            ),
            html.Br(),
            dbc.Card([
                html.P("Departamento de la Compañia: ", style={
                    "display": "inline-block",
                    "margin-top": "0.5rem",
                    "margin-bottom": "0.5rem"
                }),
                dcc.Dropdown(
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
                    value='Customer Success',
                    style={
                        "display": "inline-block",
                        "width": "90%",
                    }
                ),
                html.P("Tipo de Recurso en CRM: ", style={
                    "display": "inline-block",
                    "margin-top": "0.5rem",
                    "margin-bottom": "0.5rem"
                }),
                dcc.Dropdown(
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
                    value='contact',
                    style={
                        "display": "inline-block",
                        "width": "90%",
                    }
                )
            ], style={
                'margin': 'auto',
                'width': '50%',
                'border': "1px solid black",
                'padding': '10px'
            }),

            html.P(id='true-label', style={
                'font-size': '2.5em',
                'text-align': 'center'
            }),
            dbc.Table(
                [
                    html.Thead(html.Tr([
                        html.Th("Sección de Hulipractice"),
                        html.Th("Predicción")]))
                ] +
                [
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
                }, bordered=True,
                hover=True,
                responsive=True,
                striped=True)
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
