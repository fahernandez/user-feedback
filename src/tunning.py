import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import math
from plotly.subplots import make_subplots
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from scipy import stats

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

    fig.show("notebook")

def tunning_graph(result, combination, param, name):
    fig = px.line(result[result['combination'] == combination].sort_values(by=['repetition', param,'rank_test_score'], ascending=True).groupby(['repetition', param]).first().reset_index().rename(columns={'repetition':"Repeticiones", param:name, "mean_test_score":"Precisión media en grupo de validación"}), x="Repeticiones", y="Precisión media en grupo de validación".format(name), color=name, title="")
    for g in fig.data:
        g.update(mode='markers+lines')
    fig.show("notebook")


def complexity_graph(features, result, combination):
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
                    mode='lines+markers+text',
                    name='Entrenamiento',
                    text=res['mean_train_score'].apply(lambda w: math.floor(w * 10 ** 3) / 10 ** 3).to_frame(),
                    textposition="bottom center",
                    showlegend=count==1,
                    line=dict(color='royalblue', width=2)), row=row_count, col=col_count)

        fig.add_trace(go.Scatter(x=res[f], y=res['mean_test_score'],
                    mode='lines+markers+text',
                    name='Validación',
                    text=res['mean_test_score'].apply(lambda w: math.floor(w * 10 ** 3) / 10 ** 3).to_frame(),
                    textposition="bottom center",
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
    fig.show("notebook")

    
def get_confusion_matrix(model_pipeline, variables, response, modelRepetions):
    results_cm = {}
    for i in range(0, modelRepetions):
        X_train, X_test, y_train, y_test = train_test_split(variables, response, test_size=0.10, random_state=i)
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)
        results_cm[i] = confusion_matrix(y_test, y_pred)
       
    final_cm = [[0 for col in range(6)] for row in range(6)]
    for i in range(0, 6):
        for j in range(0, 6):
            res = []
            for k,v in results_cm.items():
                res.append(v[i,j])
            final_cm[i][j] = stats.mode(res)[0][0]
    return np.array(final_cm)


def get_mean_precision(model_pipeline, variables, response, modelRepetions):
    result = np.zeros(modelRepetions)
    for i in range(0, modelRepetions):
        # prepare the cross-validation procedure
        cv = KFold(n_splits=10, random_state=i, shuffle=True)
        # evaluate model
        scores = cross_val_score(model_pipeline, variables, response, scoring='accuracy', cv=cv, n_jobs=-1)
        result[i] = scores.mean()
    return result.mean()
    