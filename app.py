import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from src.transformation import fit_smv, predict
# https://towardsdatascience.com/how-to-use-docker-to-deploy-a-dashboard-app-on-aws-8df5fb322708

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
model = None


app.layout = html.Div([
    html.H6("Predicción automática de comentarios para HuliPractice!"),
    html.Div(["Comentario: ",
              dcc.Input(
                  id='comment',
                  value='Mejorar letra de receta',
                  type='text')]),
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
        ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div(id='my-output'),

])

@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='comment', component_property='value'),
    Input(component_id='creator-department', component_property='value'),
    Input(component_id='resource-type', component_property='value'),
)
def update_output_div(comment, creator_department, resource_type):
    return 'Output: {}'.format(predict(
        model,
        comment,
        creator_department,
        resource_type
    ))


if __name__ == '__main__':
    model = fit_smv()
    app.run_server(host='0.0.0.0', port=5050, debug=True)
