import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from sklearn.impute import SimpleImputer
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import warnings
warnings.filterwarnings("ignore")


import dash
from dash import Dash, dcc, html
import plotly.express as px
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash_mantine_components import MantineProvider

data=pd.read_csv('data/basedet_modelis_recup.csv', sep=";", dtype={"CD_POST_BIEN_PFI":"str"})

# Traitement des données de type date 
date_cols=["date_entree_defaut","date_sortie_defaut","arrete","dt_ref_pha_DET","dtHJD_def","dtHJD_prov",
           "dt_arr_last_enc_ope","dt_arr_1st_enc_ope","arrete_1strecup"]
for col in date_cols:
    data[col]=pd.to_datetime(data[col], format="%d/%m/%y")
data["DT_MEP_OPE"]=pd.to_datetime(data["DT_MEP_OPE"],format="%d%b%Y:%H:%M:%S.%f", errors='coerce')

# Filtre sur la période utilisée
data=data[(data["DT_MEP_OPE"].dt.year >= 2007) & (data["DT_MEP_OPE"].dt.year < 2017)]


data.sort_values(by=["cd_op","arrete"],ascending=True, inplace=True)


data.insert(4,"anciennete_defaut", data.groupby('cd_op').cumcount())

#, MANTINE_CSS

# --- Exemple de DataFrame fictif ---

liste_categ = ["CD_ETAT_CIVIL",	"CD_CSP_EMP1",	"CD_CSP_EMP2",	"CD_SITFAM_EMP1"]

mantine_css_url ="https://unpkg.com/@mantine/core@7.17.0/lib/index.css"


# ---------------------------------------------------------------------
# Calcul de la matrice de corrélation pour la partie numérique
# ---------------------------------------------------------------------
list_num = ["MT_PATRIM_MOB",	"MT_CHA_HORS_OPE_PFI",	"NB_CHARGE_HORS_OPE_PFI",	"MT_PATRIM_NET",	"MT_REV_PFI"]
corr_matrix = data[list_num].corr()
fig_corr = px.imshow(
    corr_matrix,
    text_auto=".2f",
    color_continuous_scale='RdBu_r',
    # title="Matrice de corrélation"
)
fig_corr.update_layout(
    paper_bgcolor='white',
    plot_bgcolor='white',
    font=dict(color='green'),
    title_font_color='green'
)

# --- Layout enveloppé dans MantineProvider ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

# ---------------------------------------------------------------------
# Layout principal avec html.Div
# ---------------------------------------------------------------------
app.layout = html.Div(
    style={
        "display": "flex",
        "minHeight": "100vh"  # Occupe toute la hauteur de la fenêtre
    },
    children=[
        # --- Barre latérale (gauche) ---
        html.Div(
            style={
                "backgroundColor": "green",
                "color": "white",
                "width": "250px",
                "padding": "20px"
            },
            children=[
                html.H4("", style={"color": "white", "margin-top":"50px"}),
                html.H6("Refonte de modèles ELBE/LGDD", style={"color": "white"}),
                html.H6("Périmètre de la Grande Clientèle", style={"color": "white"}),
                html.Hr(style={"borderColor": "white"}),
                html.P("Analyse de données", style={"color": "white"})
            ]
        ),
        # --- Contenu principal (droite) ---
        html.Div(
            style={
                "flex": "1",
                "backgroundColor": "white",
                "color": "green",
                "padding": "20px",
                "position": "relative"  # Pour positionner le logo en haut à droite
            },
            children=[
                # Logo local en haut à droite
                html.Div(
                    html.Img(src=app.get_asset_url("Groupe Crdit Agricole logo.png"),
                             style={"width": "100px", "height": "100px"}),
                    style={
                        "position": "absolute",
                        "top": "20px",
                        "right": "20px"
                    }
                ),
                # Partie 1 : Analyse des variables catégorielles
                html.H2("Analyse des variables catégorielles",
                        style={"textAlign": "center", "color": "green", "marginTop": "60px"}),
                
                html.P("Sélectionner la variable...",
                       style={"textAlign":"center", "color":"green"}),
                dcc.Dropdown(
                    id='categ-dropdown',
                    options=[{'label': cat, 'value': cat} for cat in liste_categ],
                    value=liste_categ[0],
                    placeholder="Sélectionnez variable",
                    style={
                        'width': '50%',
                        'margin': '20px auto',
                        'display': 'block'
                    }
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardBody(dcc.Graph(id='line-average-graph'))
                                ],
                                style={"margin": "10px"}
                            ),
                            md=6
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardBody(dcc.Graph(id='line-distribution-graph'))
                                ],
                                style={"margin": "10px"}
                            ),
                            md=6
                        )
                    ],
                    style={"marginTop": "30px"}
                ),
                # Partie 2 : Analyse des variables numériques
                html.H2("Analyse des variables numériques",
                        style={"textAlign": "center", "color": "green", "marginTop": "40px"}),
                dbc.Card(
                    [
                        dbc.CardHeader("Matrice de corrélation des variables numériques",
                                       style={"color": "green", "fontWeight": "bold"}),
                        dbc.CardBody(dcc.Graph(id="corr-matrix", figure=fig_corr))
                    ],
                    style={"margin-left": "200px",
                           "margin-right":"200px"}
                )
            ]
        )
    ]
)

# ---------------------------------------------------------------------
# Callback pour mettre à jour les graphiques
# ---------------------------------------------------------------------
@app.callback(
    [Output('line-average-graph', 'figure'),
     Output('line-distribution-graph', 'figure')],
    [Input('categ-dropdown', 'value')]
)
def update_graphs(selected_cat):
    # 1) Moyenne par catégorie
    data_moy_lgd = pd.crosstab(index=data[selected_cat], columns=data["anciennete_defaut"], values=data["MT_EAD_RESID"], aggfunc=np.mean).T
    fig_avg = px.line(
        data_moy_lgd,
        title=f"Moyenne de lgd réalisée par {selected_cat} suivant l'ancienneté en défaut",
        template="simple_white"
    )
    fig_avg.update_layout(
        # paper_bgcolor='white',
        # plot_bgcolor='white',
        font=dict(color='green'),
        title_font_color='green',
        xaxis=dict(color='green'),
        yaxis=dict(color='green')
    )

    # 2) Répartition (counts) dans le temps
    data_dist = pd.crosstab(index=data[selected_cat], columns=data["anciennete_defaut"],).T
    
    fig_dist = px.line(
        data_dist,
        title=f"Répartition de {selected_cat} dans le temps",
        template="simple_white"
    )
    fig_dist.update_layout(
        # paper_bgcolor='white',
        # plot_bgcolor='white',
        font=dict(color='green'),
        title_font_color='green',
        xaxis=dict(color='green'),
        yaxis=dict(color='green')
    )

    return fig_avg, fig_dist

# ---------------------------------------------------------------------
# Exécution de l'app
# ---------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)