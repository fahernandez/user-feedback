import dash
import pandas as pd
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
from src.transformation import fit_smv, predict
from src.no_supervised import clust_model_comparisson_graph, k_model_vect, \
    k_model_bin, k_model_fec, k_model_groups, db_model_vect, \
    db_model_bin, db_model_fec, db_model_noise
from src.supervised import class_model_comparisson, best_vec_model_graph, \
    tunning_graph, complexity_graph
from src.descriptive import get_desc_graph
from nltk import word_tokenize

# https://towardsdatascience.com/how-to-use-docker-to-deploy-a-dashboard-app-on-aws-8df5fb322708

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.LUX],
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
    if tab == 'no-sup':
        return html.Div([
            dbc.Tabs(id="no-sub-tabs", active_tab="no-sup-comp", children=[
                dbc.Tab(label="K-medias", tab_id="k-means"),
                dbc.Tab(label="DB-Scan", tab_id="db-scan"),
                dbc.Tab(
                    label="Comparación de técnicas de agrupación",
                    tab_id="no-sup-comp"),
            ]),
            dbc.Card(id="no-sub-content", body=True)
        ])
    elif tab == 'sup':
        return html.Div([
            dbc.Tabs(id="sub-tabs", active_tab="sup-comp", children=[
                dbc.Tab(label="Ingénuo de bayes", tab_id="bayes"),
                dbc.Tab(label="Máquinas de soporte vectorial", tab_id="svm"),
                dbc.Tab(label="Árboles aleatorios", tab_id="rf"),
                dbc.Tab(label="XGBoost", tab_id="xgboost"),
                dbc.Tab(
                    label="Comparación de técnicas de predicción",
                    tab_id="sup-comp"),
            ]),
            dbc.Card(id="sub-content", body=True)
        ])
    elif tab == 'desc':
        return html.Div(
            [
                html.Div([
                    html.Div([
                        html.P("Departamento de la compañia: ", style={
                            "margin-top": "0.5rem",
                            "margin-bottom": "0.5rem"
                        }),
                        dcc.Dropdown(
                            id='creator-department-desc',
                            options=[
                                {
                                    'label': 'Customer success',
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
                        html.P("Tipo de recurso en CRM: ", style={
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
                    'border': "1px solid rgba(0,0,0,0.125)",
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
                html.P("Departamento de la compañia: ", style={
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
                html.P("Tipo de recurso en CRM: ", style={
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
                'border': "1px solid rgba(0,0,0,0.125)",
                'padding': '10px'
            }),

            html.P(id='true-label', style={
                'font-size': '2.5em',
                'text-align': 'center'
            }),
            dbc.Table(
                [
                    html.Thead(html.Tr([
                        html.Th("Sección de hulipractice"),
                        html.Th("Predicción")]))
                ] +
                [
                    html.Tr([html.Td(['Agenda']), html.Td(id='agenda')]),
                    html.Tr([html.Td(['Consulta']), html.Td(id='checkup')]),
                    html.Tr([html.Td(['Facturación']), html.Td(id='billing')]),
                    html.Tr([html.Td(['Información médica']),
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
            'border': "1px solid rgba(0,0,0,0.125)",
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
    comments = pd.read_csv('/data-analysis/include/proccesed_comments.csv',
                           index_col='id')
    comments['tokens'] = comments.apply(
        lambda row: word_tokenize(row['comment']),
        axis=1)
    fig1, fig2, fig3, fig4 = get_desc_graph(part, creator, resource, comments)
    return fig1, fig2, fig3, fig4


@app.callback(Output('sub-content', 'children'),
              Input('sub-tabs', 'active_tab'))
def render_sub_content(tab):
    if tab == 'sup-comp':
        return html.Div([
            html.Div([
                dcc.Graph(
                    id='sub-comparison-graph',
                    figure=class_model_comparisson(),
                )
            ], style={
                'margin': 'auto',
                'width': '70%',
                'border': "1px solid rgba(0,0,0,0.125)",
                'padding': '10px'
            })
        ])
    elif tab == 'bayes':
        return html.Div([
            dbc.Tabs(id="bayes-tabs", active_tab="bayes-adj", children=[
                dbc.Tab(label="Ajuste de parámetros", tab_id="bayes-adj"),
                dbc.Tab(label="Análisis de complejidad", tab_id="bayes-complex"),
            ]),
            dbc.Card(id="bayes-content", body=True)
        ])

    elif tab == 'svm':
        return html.Div([
            dbc.Tabs(id="svm-tabs", active_tab="svm-adj", children=[
                dbc.Tab(label="Ajuste de parámetros", tab_id="svm-adj"),
                dbc.Tab(label="Análisis de complejidad", tab_id="svm-complex"),
            ]),
            dbc.Card(id="svm-content", body=True)
        ])

    elif tab == 'rf':
        return html.Div([
            dbc.Tabs(id="rf-tabs", active_tab="rf-adj", children=[
                dbc.Tab(label="Ajuste de parámetros", tab_id="rf-adj"),
                dbc.Tab(label="Análisis de complejidad", tab_id="rf-complex"),
            ]),
            dbc.Card(id="rf-content", body=True)
        ])

    elif tab == 'xgboost':
        return html.Div([
            dbc.Tabs(id="xgboost-tabs", active_tab="xgboost-adj", children=[
                dbc.Tab(label="Ajuste de parámetros", tab_id="xgboost-adj"),
                dbc.Tab(label="Análisis de complejidad", tab_id="xgboost-complex"),
            ]),
            dbc.Card(id="xgboost-content", body=True)
        ])


@app.callback(Output('xgboost-content', 'children'),
              Input('xgboost-tabs', 'active_tab'))
def render_xgboost_content(tab):
    results = pd.read_csv('/data-analysis/include/xgboost-tunning.csv')
    if tab == 'xgboost-adj':
        return html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='xgboost-vec-graph',
                        figure=best_vec_model_graph(results)
                    ),
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
                html.Div([
                    dcc.Graph(
                        id='xgboost-fec-graph',
                        figure=tunning_graph(results, "bow", "tokens",
                                             "Features")
                    )
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='xgboost-est-graph',
                        figure=tunning_graph(results, "bow", "n_estimators",
                                             "Regresores")
                    ),
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
                html.Div([
                    dcc.Graph(
                        id='xgboost-ngram-graph',
                        figure=tunning_graph(results, "bow", "ngram_range",
                                             "Ngram")
                    )
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='xgboost-bin-graph',
                        figure=tunning_graph(results, "bow", "binary",
                                             "Binarización")
                    ),
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
                html.Div([
                    dcc.Graph(
                        id='CCP Alpha',
                        figure=tunning_graph(results, "bow", "ccp_alpha",
                                             "CCP Alpha")
                    )
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                })
            ])
        ])

    elif tab == 'xgboost-complex':
        features = {
            'ccp_alpha': "CCP Alpha",
            'tokens': "Features",
            'n_estimators': 'Regresores'

        }
        return html.Div([
            html.Div([
                dcc.Graph(
                    id='xgboost-complex-graph',
                    figure=complexity_graph(features, results, 'bow', 0.75),
                    style={
                        'height': '900px'
                    }
                ),
            ], style={

            })
        ])


@app.callback(Output('rf-content', 'children'),
              Input('rf-tabs', 'active_tab'))
def render_rf_content(tab):
    results = pd.read_csv('/data-analysis/include/forest-tunning.csv')
    if tab == 'rf-adj':
        return html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='rf-vec-graph',
                        figure=best_vec_model_graph(results)
                    ),
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
                html.Div([
                    dcc.Graph(
                        id='rf-fec-graph',
                        figure=tunning_graph(results, "bow", "tokens",
                                             "Features")
                    )
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='rf-est-graph',
                        figure=tunning_graph(results, "bow", "n_estimators",
                                             "Arboles")
                    ),
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
                html.Div([
                    dcc.Graph(
                        id='rf-ngram-graph',
                        figure=tunning_graph(results, "bow", "ngram_range",
                                             "Features")
                    )
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='rf-bin-graph',
                        figure=tunning_graph(results, "bow", "binary",
                                             "Binarización")
                    ),
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
                html.Div([
                    dcc.Graph(
                        id='rf-par-complex-graph',
                        figure=tunning_graph(results, "bow", "ccp_alpha",
                                             "Parámetro de complejidad")
                    )
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                })
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='rf-max-graph',
                        figure=tunning_graph(results, "bow", "max_features",
                                             "M")
                    ),
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                })
            ])
        ])

    elif tab == 'rf-complex':
        features = {
            'n_estimators':  "Arboles",
            'tokens': "Features",
            'ccp_alpha': "Parámetro de Complejidad"
        }

        return html.Div([
            html.Div([
                dcc.Graph(
                    id='rf-complex-graph',
                    figure=complexity_graph(features, results, 'bow', 0.75),
                    style={
                        'height': '900px'
                    }
                ),
            ], style={

            })
        ])


@app.callback(Output('svm-content', 'children'),
              Input('svm-tabs', 'active_tab'))
def render_svm_content(tab):
    results = pd.read_csv('/data-analysis/include/svc-tunning.csv')
    if tab == 'svm-adj':
        return html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='svm-vec-graph',
                        figure=best_vec_model_graph(results)
                    ),
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
                html.Div([
                    dcc.Graph(
                        id='svm-fec-graph',
                        figure=tunning_graph(results, "tf-idf", "tokens",
                                             "Features")
                    )
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='svm-gamma-graph',
                        figure=tunning_graph(results, "tf-idf", "gamma",
                                             "Gamma")
                    ),
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
                html.Div([
                    dcc.Graph(
                        id='svm-ngram-graph',
                        figure=tunning_graph(results, "tf-idf", "ngram_range",
                                             "Ngram")
                    )
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='svm-bin-graph',
                        figure=tunning_graph(results, "tf-idf", "binary",
                                             "Binarización")
                    ),
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
                html.Div([
                    dcc.Graph(
                        id='svm-c-graph',
                        figure=tunning_graph(results, "tf-idf", "C", "C")
                    )
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                })
            ])
        ])

    elif tab == 'svm-complex':
        features = {
            'C': 'C',
            'tokens': "Features"
        }
        return html.Div([
            html.Div([
                dcc.Graph(
                    id='svm-complex-graph',
                    figure=complexity_graph(features, results, 'tf-idf', 0.77)
                ),
            ], style={
                'display': 'inline-block',
                'width': '100%'
            })
        ])


@app.callback(Output('bayes-content', 'children'),
              Input('bayes-tabs', 'active_tab'))
def render_bayes_content(tab):
    results = pd.read_csv('/data-analysis/include/bayes-tunning.csv')
    if tab == 'bayes-adj':
        return html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='bayes-vec-graph',
                        figure=best_vec_model_graph(results)
                    ),
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
                html.Div([
                    dcc.Graph(
                        id='bayes-fec-graph',
                        figure=tunning_graph(results, "bow", "tokens",
                                             "Features")
                    )
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='bayes-alpha-graph',
                        figure=tunning_graph(results, "bow", "alpha",
                                             "Alpha")
                    ),
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
                html.Div([
                    dcc.Graph(
                        id='bayes-ngram-graph',
                        figure=tunning_graph(results, "bow", "ngram_range",
                                             "Ngram")
                    )
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='bayes-bin-graph',
                        figure=tunning_graph(results, "bow", "binary",
                                             "Binarización")
                    ),
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
            ])
        ])

    elif tab == 'bayes-complex':
        features = {
            'alpha': "Alpha",
            'tokens': "Features"
        }
        return html.Div([
            html.Div([
                dcc.Graph(
                    id='bayes-complex-graph',
                    figure=complexity_graph(features, results, 'bow', 0.77)
                ),
            ], style={
                'display': 'inline-block',
                'width': '100%'
            }),
        ])


@app.callback(Output('no-sub-content', 'children'),
              Input('no-sub-tabs', 'active_tab'))
def render_no_sub_content(tab):
    if tab == 'no-sup-comp':
        return html.Div([
            html.Div([
                dcc.Graph(
                    id='no-sub-comparison-graph',
                    figure=clust_model_comparisson_graph(),
                )
            ], style={
                'display': 'inline-block',
                'width': '49%'
            }),
            html.Div([
                html.Label("Agrupación de comentarios según mejores parámetros K-Medias",
                           style={
                               "margin-bottom": "3.5rem"
                           }),
                dbc.Table(
                    [
                        html.Thead(html.Tr([
                            html.Th("Categoría"),
                            html.Th("Grupo 1"),
                            html.Th("Grupo 2"),
                            html.Th("Grupo 3"),
                            html.Th("Grupo 4"),
                            html.Th("Grupo 5"),
                        ]))
                    ] +
                    [
                        html.Tr(
                            [
                                html.Td(['Inf. médica']),
                                html.Td(['4']),
                                html.Td(['27']),
                                html.Td(['45']),
                                html.Td(['93']),
                                html.Td(['166']),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td(['Agenda']),
                                html.Td(['6']),
                                html.Td(['94']),
                                html.Td(['9']),
                                html.Td(['232']),
                                html.Td(['1']),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td(['Facturacíon']),
                                html.Td(['211']),
                                html.Td(['30']),
                                html.Td(['42']),
                                html.Td(['39']),
                                html.Td(['16']),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td(['Consulta']),
                                html.Td(['1']),
                                html.Td(['1']),
                                html.Td(['59']),
                                html.Td(['16']),
                                html.Td(['149']),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td(['Servicio']),
                                html.Td(['3']),
                                html.Td(['38']),
                                html.Td(['64']),
                                html.Td(['64']),
                                html.Td(['27']),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td(['Perfil']),
                                html.Td(['0']),
                                html.Td(['28']),
                                html.Td(['8']),
                                html.Td(['17']),
                                html.Td(['22']),
                            ]
                        )
                    ],
                    bordered=True,
                    hover=True,
                    responsive=True,
                    striped=True
                )], style={
                'display': 'inline-block',
                'width': '49%'
            }),
        ])
    elif tab == 'k-means':
        return html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='no-sup-k-vec-graph',
                        figure=k_model_vect()
                    ),
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
                html.Div([
                    dcc.Graph(
                        id='no-sup-k-bi-graph',
                        figure=k_model_bin()
                    )
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='no-sup-k-fec-graph',
                        figure=k_model_fec()
                    ),
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
                html.Div([
                    dcc.Graph(
                        id='no-sup-k-groups-graph',
                        figure=k_model_groups()
                    )
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
            ]),
        ])

    elif tab == 'db-scan':
        return html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='no-sup-db-vec-graph',
                        figure=db_model_vect()
                    ),
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
                html.Div([
                    dcc.Graph(
                        id='no-sup-db-bi-graph',
                        figure=db_model_bin()
                    )
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='no-sup-db-fec-graph',
                        figure=db_model_fec()
                    ),
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
                html.Div([
                    dcc.Graph(
                        id='no-sup-db-fec-graph',
                        figure=db_model_noise()
                    ),
                ], style={
                    'display': 'inline-block',
                    'width': '49%'
                }),
            ]),
        ])


if __name__ == '__main__':
    model = fit_smv()
    app.run_server(host='0.0.0.0', port=5050, debug=True)
