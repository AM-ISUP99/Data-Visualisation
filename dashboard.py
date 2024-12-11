import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from streamlit_option_menu import option_menu
import io
import json 
import base64
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from keplergl import KeplerGl
import geopandas as gpd

# Configuration de la page
st.set_page_config(
    page_title="Life Expectancy Dashboard",
    page_icon="🌍",
    layout="wide",

)

# Chargement des données
@st.cache_data  # Cette décoration permet de mettre en cache les données
def load_data():
    try:
        # Remplacez 'votre_fichier.csv' par le nom de votre fichier
        df = pd.read_csv('data/mon_fichier.csv')
        df['Status'] = df.apply(
        lambda row: 'Developed' if row['Economy_status_Developed'] == 1 else 'Developing',
        axis=1
        )
    
        # Supprimer les colonnes de statut économique binaires car elles ne sont plus nécessaires
        df = df.drop(['Economy_status_Developed', 'Economy_status_Developing'], axis=1)
    
        return df
    except FileNotFoundError:
        st.error("Fichier de données non trouvé!")
        # Retourner des données factices en cas d'erreur
        return pd.DataFrame()  # ou vos données exemple actuelles

# Chargement des données
df = load_data()

def create_kepler_map():
    st.markdown('<div class="gradient-text">Carte interactive</div>', unsafe_allow_html=True)
    
    @st.cache_data
    def load_geo_data():
        # Charger les données géographiques des pays
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

        # Liste complète des colonnes numériques
        numeric_columns = [
            'Life_expectancy', 'Adult_mortality', 'Infant_deaths', 
            'Alcohol_consumption', 'Hepatitis_B', 'Measles',
            'BMI', 'Under_five_deaths', 'Polio', 'GDP_per_capita',
            'Population_mln', 'Thinness_ten_nineteen_years',
            'Thinness_five_nine_years', 'Diphtheria',
            'Incidents_HIV', 'Schooling'
        ]
        
        # Définir le dictionnaire de mapping
        country_name_mapping = {
            'United States of America': 'United States',
            'Dominican Rep.': 'Dominican Republic',
            'Central African Rep.': 'Central African Republic',
            'Dem. Rep. Congo': 'Democratic Republic of the Congo',
            'Congo': 'Republic of Congo',
            'S. Sudan': 'South Sudan',
            'Solomon Is.': 'Solomon Islands',
            'Eq. Guinea': 'Equatorial Guinea',
            'Guinea-Bissau': 'Guinea Bissau',
            'Bosnia and Herz.': 'Bosnia and Herzegovina',
            'Czech Rep.': 'Czech Republic',
            'Macedonia': 'North Macedonia',
            'Slovakia': 'Slovak Republic',
            'Brunei': 'Brunei Darussalam',
            'Timor-Leste': 'East Timor',
            'Lao PDR': 'Laos',
            'Vietnam': 'Viet Nam'
        }

        # Appliquer le mapping
        world['name'] = world['name'].replace(country_name_mapping)

        # Obtenir les dernières données
        df_latest = df[df['Year'] == df['Year'].max()]

        # Créer le GeoDataFrame
        geo_df = gpd.GeoDataFrame(
            world.merge(df_latest, how='left', left_on=['name'], right_on=['Country']),
            geometry='geometry'
        )

        # Convertir toutes les colonnes en numérique
        for col in numeric_columns:
            if col in geo_df.columns:
                geo_df[col] = pd.to_numeric(geo_df[col], errors='coerce')

        # Remplir les valeurs manquantes
        for col in numeric_columns:
            if col in geo_df.columns:
                geo_df[col].fillna(df_latest[col].mean(), inplace=True)
        
        return geo_df

    # Charger les données
    geo_df = load_geo_data()

    # Configuration de Kepler
    config = {'version': 'v1', 'config': {'visState': {'filters': [], 'layers': [{'id': 'gsk43pu', 'type': 'geojson', 'config': {'dataId': 'data_1', 'label': 'Espérance de vie', 'color': [241, 92, 23], 'highlightColor': [252, 242, 26, 255], 'columns': {'geojson': 'geometry'}, 'isVisible': True, 'visConfig': {'opacity': 0.8, 'strokeOpacity': 0.8, 'thickness': 0.5, 'strokeColor': [136, 87, 44], 'colorRange': {'name': 'ColorBrewer RdYlGn-6', 'type': 'diverging', 'category': 'ColorBrewer', 'colors': ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']}, 'strokeColorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber', 'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']}, 'radius': 10, 'sizeRange': [0, 10], 'radiusRange': [0, 50], 'heightRange': [0, 500], 'elevationScale': 5, 'enableElevationZoomFactor': True, 'stroked': True, 'filled': True, 'enable3d': False, 'wireframe': False}, 'hidden': False, 'textLabel': [{'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start', 'alignment': 'center'}]}, 'visualChannels': {'colorField': {'name': 'Life_expectancy', 'type': 'real'}, 'colorScale': 'quantize', 'strokeColorField': None, 'strokeColorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear', 'heightField': None, 'heightScale': 'linear', 'radiusField': None, 'radiusScale': 'linear'}}, {'id': 'xvagl3n', 'type': 'geojson', 'config': {'dataId': 'data_1', 'label': 'Mortalité infantile ', 'color': [241, 92, 23], 'highlightColor': [252, 242, 26, 255], 'columns': {'geojson': 'geometry'}, 'isVisible': False, 'visConfig': {'opacity': 0.8, 'strokeOpacity': 0.8, 'thickness': 0.5, 'strokeColor': [136, 87, 44], 'colorRange': {'name': 'Uber Viz Diverging 1.5', 'type': 'diverging', 'category': 'Uber', 'colors': ['#00939C', '#5DBABF', '#BAE1E2', '#F8C0AA', '#DD7755', '#C22E00']}, 'strokeColorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber', 'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']}, 'radius': 10, 'sizeRange': [0, 10], 'radiusRange': [0, 50], 'heightRange': [0, 500], 'elevationScale': 5, 'enableElevationZoomFactor': True, 'stroked': True, 'filled': True, 'enable3d': False, 'wireframe': False}, 'hidden': False, 'textLabel': [{'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start', 'alignment': 'center'}]}, 'visualChannels': {'colorField': {'name': 'Infant_deaths', 'type': 'real'}, 'colorScale': 'quantize', 'strokeColorField': None, 'strokeColorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear', 'heightField': None, 'heightScale': 'linear', 'radiusField': None, 'radiusScale': 'linear'}}, {'id': 'nn7tyc', 'type': 'geojson', 'config': {'dataId': 'data_1', 'label': 'Mortalité adulte', 'color': [241, 92, 23], 'highlightColor': [252, 242, 26, 255], 'columns': {'geojson': 'geometry'}, 'isVisible': False, 'visConfig': {'opacity': 0.8, 'strokeOpacity': 0.8, 'thickness': 0.5, 'strokeColor': [136, 87, 44], 'colorRange': {'name': 'Uber Viz Diverging 1.5', 'type': 'diverging', 'category': 'Uber', 'colors': ['#00939C', '#5DBABF', '#BAE1E2', '#F8C0AA', '#DD7755', '#C22E00']}, 'strokeColorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber', 'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']}, 'radius': 10, 'sizeRange': [0, 10], 'radiusRange': [0, 50], 'heightRange': [0, 500], 'elevationScale': 5, 'enableElevationZoomFactor': True, 'stroked': True, 'filled': True, 'enable3d': False, 'wireframe': False}, 'hidden': False, 'textLabel': [{'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start', 'alignment': 'center'}]}, 'visualChannels': {'colorField': {'name': 'Adult_mortality', 'type': 'real'}, 'colorScale': 'quantize', 'strokeColorField': None, 'strokeColorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear', 'heightField': None, 'heightScale': 'linear', 'radiusField': None, 'radiusScale': 'linear'}}, {'id': '8kg1roa', 'type': 'geojson', 'config': {'dataId': 'data_1', 'label': 'BMI', 'color': [241, 92, 23], 'highlightColor': [252, 242, 26, 255], 'columns': {'geojson': 'geometry'}, 'isVisible': False, 'visConfig': {'opacity': 0.8, 'strokeOpacity': 0.8, 'thickness': 0.5, 'strokeColor': [136, 87, 44], 'colorRange': {'name': 'ColorBrewer YlGn-6', 'type': 'sequential', 'category': 'ColorBrewer', 'colors': ['#ffffcc', '#d9f0a3', '#addd8e', '#78c679', '#31a354', '#006837']}, 'strokeColorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber', 'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']}, 'radius': 10, 'sizeRange': [0, 10], 'radiusRange': [0, 50], 'heightRange': [0, 500], 'elevationScale': 5, 'enableElevationZoomFactor': True, 'stroked': True, 'filled': True, 'enable3d': False, 'wireframe': False}, 'hidden': False, 'textLabel': [{'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start', 'alignment': 'center'}]}, 'visualChannels': {'colorField': {'name': 'BMI', 'type': 'real'}, 'colorScale': 'quantize', 'strokeColorField': None, 'strokeColorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear', 'heightField': None, 'heightScale': 'linear', 'radiusField': None, 'radiusScale': 'linear'}}, {'id': 'k7upy2t', 'type': 'geojson', 'config': {'dataId': 'data_1', 'label': 'GDP per capita', 'color': [241, 92, 23], 'highlightColor': [252, 242, 26, 255], 'columns': {'geojson': 'geometry'}, 'isVisible': False, 'visConfig': {'opacity': 0.8, 'strokeOpacity': 0.8, 'thickness': 0.5, 'strokeColor': [136, 87, 44], 'colorRange': {'name': 'ColorBrewer RdBu-6', 'type': 'diverging', 'category': 'ColorBrewer', 'colors': ['#b2182b', '#ef8a62', '#fddbc7', '#d1e5f0', '#67a9cf', '#2166ac']}, 'strokeColorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber', 'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']}, 'radius': 10, 'sizeRange': [0, 10], 'radiusRange': [0, 50], 'heightRange': [0, 500], 'elevationScale': 5, 'enableElevationZoomFactor': True, 'stroked': True, 'filled': True, 'enable3d': False, 'wireframe': False}, 'hidden': False, 'textLabel': [{'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start', 'alignment': 'center'}]}, 'visualChannels': {'colorField': {'name': 'GDP_per_capita', 'type': 'real'}, 'colorScale': 'quantize', 'strokeColorField': None, 'strokeColorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear', 'heightField': None, 'heightScale': 'linear', 'radiusField': None, 'radiusScale': 'linear'}}, {'id': 'n1csad6', 'type': 'geojson', 'config': {'dataId': 'data_1', 'label': 'Population', 'color': [241, 92, 23], 'highlightColor': [252, 242, 26, 255], 'columns': {'geojson': 'geometry'}, 'isVisible': False, 'visConfig': {'opacity': 0.8, 'strokeOpacity': 0.8, 'thickness': 0.5, 'strokeColor': [136, 87, 44], 'colorRange': {'name': 'ColorBrewer RdYlGn-6', 'type': 'diverging', 'category': 'ColorBrewer', 'colors': ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59', '#d73027'], 'reversed': True}, 'strokeColorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber', 'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']}, 'radius': 10, 'sizeRange': [0, 10], 'radiusRange': [0, 50], 'heightRange': [0, 500], 'elevationScale': 5, 'enableElevationZoomFactor': True, 'stroked': True, 'filled': True, 'enable3d': False, 'wireframe': False}, 'hidden': False, 'textLabel': [{'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start', 'alignment': 'center'}]}, 'visualChannels': {'colorField': {'name': 'Population_mln', 'type': 'real'}, 'colorScale': 'quantize', 'strokeColorField': None, 'strokeColorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear', 'heightField': None, 'heightScale': 'linear', 'radiusField': None, 'radiusScale': 'linear'}}, {'id': 'vzla5gt', 'type': 'geojson', 'config': {'dataId': 'data_1', 'label': 'VIH', 'color': [241, 92, 23], 'highlightColor': [252, 242, 26, 255], 'columns': {'geojson': 'geometry'}, 'isVisible': False, 'visConfig': {'opacity': 0.8, 'strokeOpacity': 0.8, 'thickness': 0.5, 'strokeColor': [136, 87, 44], 'colorRange': {'name': 'ColorBrewer YlOrRd-6', 'type': 'sequential', 'category': 'ColorBrewer', 'colors': ['#ffffb2', '#fed976', '#feb24c', '#fd8d3c', '#f03b20', '#bd0026'], 'reversed': False}, 'strokeColorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber', 'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']}, 'radius': 10, 'sizeRange': [0, 10], 'radiusRange': [0, 50], 'heightRange': [0, 500], 'elevationScale': 5, 'enableElevationZoomFactor': True, 'stroked': True, 'filled': True, 'enable3d': False, 'wireframe': False}, 'hidden': False, 'textLabel': [{'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start', 'alignment': 'center'}]}, 'visualChannels': {'colorField': {'name': 'Incidents_HIV', 'type': 'real'}, 'colorScale': 'quantize', 'strokeColorField': None, 'strokeColorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear', 'heightField': None, 'heightScale': 'linear', 'radiusField': None, 'radiusScale': 'linear'}}, {'id': '8e625fv', 'type': 'geojson', 'config': {'dataId': 'data_1', 'label': 'Hépatite B', 'color': [241, 92, 23], 'highlightColor': [252, 242, 26, 255], 'columns': {'geojson': 'geometry'}, 'isVisible': False, 'visConfig': {'opacity': 0.8, 'strokeOpacity': 0.8, 'thickness': 0.5, 'strokeColor': [136, 87, 44], 'colorRange': {'name': 'ColorBrewer RdYlGn-6', 'type': 'diverging', 'category': 'ColorBrewer', 'colors': ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']}, 'strokeColorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber', 'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']}, 'radius': 10, 'sizeRange': [0, 10], 'radiusRange': [0, 50], 'heightRange': [0, 500], 'elevationScale': 5, 'enableElevationZoomFactor': True, 'stroked': True, 'filled': True, 'enable3d': False, 'wireframe': False}, 'hidden': False, 'textLabel': [{'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start', 'alignment': 'center'}]}, 'visualChannels': {'colorField': {'name': 'Hepatitis_B', 'type': 'real'}, 'colorScale': 'quantize', 'strokeColorField': None, 'strokeColorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear', 'heightField': None, 'heightScale': 'linear', 'radiusField': None, 'radiusScale': 'linear'}}, {'id': '3ko6icb', 'type': 'geojson', 'config': {'dataId': 'data_1', 'label': 'Éducation ', 'color': [241, 92, 23], 'highlightColor': [252, 242, 26, 255], 'columns': {'geojson': 'geometry'}, 'isVisible': False, 'visConfig': {'opacity': 0.8, 'strokeOpacity': 0.8, 'thickness': 0.5, 'strokeColor': [136, 87, 44], 'colorRange': {'name': 'ColorBrewer RdYlGn-6', 'type': 'diverging', 'category': 'ColorBrewer', 'colors': ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']}, 'strokeColorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber', 'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']}, 'radius': 10, 'sizeRange': [0, 10], 'radiusRange': [0, 50], 'heightRange': [0, 500], 'elevationScale': 5, 'enableElevationZoomFactor': True, 'stroked': True, 'filled': True, 'enable3d': False, 'wireframe': False}, 'hidden': False, 'textLabel': [{'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start', 'alignment': 'center'}]}, 'visualChannels': {'colorField': {'name': 'Schooling', 'type': 'real'}, 'colorScale': 'quantize', 'strokeColorField': None, 'strokeColorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear', 'heightField': None, 'heightScale': 'linear', 'radiusField': None, 'radiusScale': 'linear'}}], 'interactionConfig': {'tooltip': {'fieldsToShow': {'data_1': [{'name': 'name', 'format': None}, {'name': 'Population_mln', 'format': None}, {'name': 'GDP_per_capita', 'format': None}, {'name': 'Life_expectancy', 'format': None}, {'name': 'Adult_mortality', 'format': None}, {'name': 'Infant_deaths', 'format': None}, {'name': 'Hepatitis_B', 'format': None}, {'name': 'Measles', 'format': None}, {'name': 'BMI', 'format': None}, {'name': 'Incidents_HIV', 'format': None}]}, 'compareMode': False, 'compareType': 'absolute', 'enabled': True}, 'brush': {'size': 0.5, 'enabled': False}, 'geocoder': {'enabled': False}, 'coordinate': {'enabled': False}}, 'layerBlending': 'normal', 'splitMaps': [], 'animationConfig': {'currentTime': None, 'speed': 1}}, 'mapState': {'bearing': 0, 'dragRotate': False, 'latitude': 45.89210339076884, 'longitude': 15.436628808912463, 'pitch': 0, 'zoom': 1.3245626891605111, 'isSplit': False}, 'mapStyle': {'styleType': 'dark', 'topLayerGroups': {}, 'visibleLayerGroups': {'label': True, 'road': True, 'border': False, 'building': True, 'water': True, 'land': True, '3d building': False}, 'threeDBuildingColor': [9.665468314072013, 17.18305478057247, 31.1442867897876], 'mapStyles': {}}}}

    # Créer la carte
    map_1 = KeplerGl(height=600, config=config)
    
    # Ajouter les données
    map_1.add_data(data=geo_df, name="data_1")
    
    # Afficher
    st.components.v1.html(map_1._repr_html_(), height=600)

    # Après l'affichage de la carte, ajouter le guide
    with st.expander("📖 Guide d'utilisation de la carte", expanded=False):
        st.markdown("""
        ### Navigation de base
        - 🖱️ **Déplacement** : Cliquez et faites glisser la carte
        - 🔍 **Zoom** : Utilisez la molette de la souris ou le pavé tactile
        - 🌍 **Rotation** : Maintenez SHIFT + cliquez et faites glisser
        - 🔄 **Réinitialiser la vue** : Double-cliquez sur la carte
        
        ### Interactions avancées
        - 📍 **Informations sur un pays** : Cliquez sur un pays
        - 📊 **Légende** : En bas à droite
        - ⚙️ **Menu des paramètres** : Icône en haut à droite
        
        ### Panneau de configuration
        - 🎨 **Layers** : Apparence de la carte
        - 🎯 **Filters** : Filtrage des données
        - 📝 **Interactions** : Paramètres d'interaction
        - 💾 **Base Map** : Style de la carte
        
        ### Code couleur
        - 🔴 Rouge foncé : Espérance de vie plus faible
        - 🟡 Jaune : Espérance de vie moyenne
        - 🟢 Vert : Espérance de vie plus élevée
        
        ### Astuces
        - 💡 Ajustez l'opacité pour une meilleure visibilité
        - 📏 Échelle disponible en bas à gauche
        - 📊 Données basées sur les statistiques les plus récentes
        - 🔍 Utilisez les filtres pour des analyses ciblées
        """)

def prepare_prediction_model(df):
    """
    Prépare le modèle de prédiction en sélectionnant les variables les plus importantes.
    """
    # Copie du DataFrame pour éviter les modifications sur l'original
    df_model = df.copy()
    
    # Sélection initiale des features numériques (excluding target and non-predictive columns)
    feature_columns = df_model.select_dtypes(include=['float64', 'int64']).columns.tolist()
    feature_columns = [col for col in feature_columns if col not in ['Life_expectancy', 'Year']]
    
    # Préparation des données
    X = df_model[feature_columns]
    y = df_model['Life_expectancy']
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardisation des features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Création et entraînement du modèle pour la sélection de features
    sel_model = RandomForestRegressor(n_estimators=100, random_state=42)
    sel_model.fit(X_train_scaled, y_train)
    
    # Sélection des features les plus importantes
    selector = SelectFromModel(sel_model, prefit=True)
    feature_mask = selector.get_support()
    selected_features = [feature for feature, selected in zip(feature_columns, feature_mask) if selected]
    
    # Création et entraînement du modèle final avec les features sélectionnées
    final_model = RandomForestRegressor(n_estimators=100, random_state=42)
    final_model.fit(X_train[selected_features], y_train)
    
    return final_model, selected_features, scaler

def calculate_trend(data, column):
    """
    Calcule la tendance moyenne annuelle pour une colonne donnée
    """
    if len(data) < 2:
        return 0
    
    yearly_changes = []
    values = data[column].values
    
    for i in range(1, len(values)):
        if values[i-1] != 0:  # Éviter la division par zéro
            yearly_change = (values[i] - values[i-1]) / values[i-1]
            yearly_changes.append(yearly_change)
    
    if not yearly_changes:
        return 0
    
    # Retourne la médiane des changements pour être plus robuste aux valeurs extrêmes
    return np.median(yearly_changes)

def prepare_prediction_model(df):
    """
    Prépare le modèle de prédiction en sélectionnant les variables les plus importantes.
    """
    # Copie du DataFrame pour éviter les modifications sur l'original
    df_model = df.copy()
    
    # Sélection initiale des features numériques
    feature_columns = df_model.select_dtypes(include=['float64', 'int64']).columns.tolist()
    feature_columns = [col for col in feature_columns if col not in ['Life_expectancy', 'Year']]
    
    # Préparation des données
    X = df_model[feature_columns]
    y = df_model['Life_expectancy']
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardisation des features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Création et entraînement du modèle pour la sélection de features
    sel_model = RandomForestRegressor(n_estimators=100, random_state=42)
    sel_model.fit(X_train_scaled, y_train)
    
    # Sélection des features les plus importantes
    selector = SelectFromModel(sel_model, prefit=True, max_features=10)  # Limiter à 10 features
    feature_mask = selector.get_support()
    selected_features = [feature for feature, selected in zip(feature_columns, feature_mask) if selected]
    
    # Création et entraînement du modèle final avec les features sélectionnées
    final_model = RandomForestRegressor(n_estimators=100, random_state=42)
    final_model.fit(X_train[selected_features], y_train)
    
    # Calcul de l'importance des features
    feature_importance = dict(zip(selected_features, final_model.feature_importances_))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    return final_model, selected_features, scaler, sorted_features

def calculate_trend(data, column):
    """
    Calcule la tendance moyenne annuelle pour une colonne donnée
    en se concentrant sur les dernières années pour mieux capturer
    la tendance récente
    """
    if len(data) < 2:
        return 0
    
    # Se concentrer sur les 5 dernières années pour la tendance
    recent_data = data.tail(5)
    if len(recent_data) < 2:
        recent_data = data
    
    yearly_changes = []
    values = recent_data[column].values
    
    for i in range(1, len(values)):
        if values[i-1] != 0:  # Éviter la division par zéro
            yearly_change = (values[i] - values[i-1]) / values[i-1]
            yearly_changes.append(yearly_change)
    
    if not yearly_changes:
        return 0
    
    return np.mean(yearly_changes)

def predict_life_expectancy(model, features, scaler, country_data, years_ahead):
    """
    Prédit l'espérance de vie avec une seed fixe pour la reproductibilité
    """
    # Fixer la seed pour numpy
    np.random.seed(42)
    
    predictions = []
    current_data = country_data[features].iloc[-1:].copy()
    
    # Obtenir la dernière valeur connue
    last_known_value = country_data['Life_expectancy'].iloc[-1]
    
    # Calculer la tendance moyenne annuelle sur les 5 dernières années
    recent_data = country_data.sort_values('Year').tail(5)
    yearly_changes = recent_data['Life_expectancy'].diff().dropna()
    avg_yearly_increase = yearly_changes.mean()
    
    # Générer les prédictions
    current_value = last_known_value
    
    for year in range(years_ahead):
        # Ajouter une petite variabilité à la tendance
        variation = np.random.normal(0, abs(avg_yearly_increase) * 0.1)  # 10% de variabilité
        increase = avg_yearly_increase + variation
        
        # Réduire progressivement l'augmentation à mesure qu'on approche du maximum
        max_life_expectancy = 95  # Valeur maximum réaliste
        progress_to_max = (current_value / max_life_expectancy)
        reduction_factor = max(0, 1 - progress_to_max**2)
        increase = increase * reduction_factor
        
        # Calculer la nouvelle valeur
        new_value = current_value + increase
        
        # S'assurer que la valeur reste dans des limites raisonnables
        new_value = min(max(new_value, current_value), max_life_expectancy)
        
        predictions.append(new_value)
        current_value = new_value
    
    # Réinitialiser la seed pour ne pas affecter d'autres parties du code
    np.random.seed(None)
    
    return predictions

# Ajouter ce style CSS global après le chargement des données
st.markdown("""
    <style>
    .gradient-text {
        background: linear-gradient(45deg, #1e88e5, #00acc1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1em;
        animation: gradient 3s ease infinite;
    }
    
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    </style>
""", unsafe_allow_html=True)

def get_variable_groups():
    """
    Retourne un dictionnaire des variables groupées par catégorie avec les noms corrects du dataset.
    """
    return {
        "Santé générale": {
            "variables": ['Life_expectancy', 'Adult_mortality', 'BMI', 'Infant_deaths', 'Under_five_deaths'],
            "description": "Indicateurs généraux de santé (Représente les décès pour 1000 habitants)",
            "max_scale": 100
        },
        "Maladies et vaccinations (%)": {
            "variables": ['Hepatitis_B', 'Polio', 'Diphtheria', 'Measles'],
            "description": "Taux de vaccination et prévalence des maladies",
            "max_scale": 100
        },
        "Facteurs de risque": {
            "variables": ['Thinness_ten_nineteen_years', 'Thinness_five_nine_years'],
            "description": "Facteurs de risque pour la santé : Prévalence de la maigreur chez les enfants",
            "max_scale": 100
        },
        "Indicateurs de développement": {
            "variables": ['Schooling'],
            "description": "Temps moyen passé à l'école",
            "max_scale": 100
        },
        "Indicateurs économiques": {
            "variables": ['GDP_per_capita'],
            "description": "Produit intérieur brut par habitant",
            "format": "currency"
        },
        "Données démographiques": {
            "variables": ['Population_mln'],
            "description": "Statistiques de population",
            "format": "large_number"
        },
        "Autres": {
            "variables": ["Incidents_HIV", "Alcohol_consumption"],
            "description": "Statistiques autres : Incidents de VIH pour 1000 habitants et consommation d'alcool en litre",
            "max_scale": 50
        }
    }

def create_comparison_chart(df_filtered, pays1, pays2, groupe, variables):
    """
    Crée un graphique de comparaison pour un groupe de variables spécifique.
    """
    data_pays1 = df_filtered[df_filtered['Country'] == pays1].iloc[0]  # Notez le changement ici pour Country_Name
    data_pays2 = df_filtered[df_filtered['Country'] == pays2].iloc[0]  # Notez le changement ici pour Country_Name
    
    # Créer un DataFrame pour la comparaison
    comparison_data = pd.DataFrame({
        'Variable': variables,
        pays1: [data_pays1[col] for col in variables],
        pays2: [data_pays2[col] for col in variables]
    })
    
    # Créer le graphique
    fig = px.bar(
        comparison_data,
        x='Variable',
        y=[pays1, pays2],
        barmode='group',
        title=f"{groupe}",
        color_discrete_sequence=['#1f77b4', '#ff7f0e']
    )
    
    # Personnaliser le graphique
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        template="plotly_white",
        showlegend=True,
        legend_title="Pays"
    )
    
    return fig

def format_value(value, format_type=None):
    """
    Formate les valeurs selon leur type.
    """
    if pd.isna(value):
        return "N/A"
    
    if format_type == "currency":
        return f"${value:,.2f}"
    elif format_type == "large_number":
        return f"{value:,.0f}"
    else:
        return f"{value:.2f}"

# Création du menu latéral avec icônes
with st.sidebar:
    # Ajout du titre avant le menu
    st.markdown("""
        <style>
        .sidebar-title {
            background: linear-gradient(45deg, #1e88e5, #00acc1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2em;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1em;
            animation: gradient 3s ease infinite;
        }
        </style>
        
        <div class="sidebar-title">
            Life Expectancy Dashboard
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)  # Ligne de séparation
    selected = option_menu(
        menu_title="Menu",
        options=["Accueil","Comparaison de pays","Évolution dans le temps", "Prédictions", "Analyses","Carte","À propos"],
        icons=["house","bar-chart", "bar-chart","graph-up", "info-circle","map"],
        menu_icon="",
        default_index=0,
    )
    # Ajout d'une ligne de séparation
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Logo et sous-titre sous le menu
    st.markdown(
        f"""
        <div style='text-align: center;'>
            <img src='data:image/jpeg;base64,{base64.b64encode(open("Logo-ISUP.png", "rb").read()).decode()}' style='width: 150px;'>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown("<p style='text-align: center; font-size: 1em;'>Projet Data Visualisation<br>ISUP 2024</p>", unsafe_allow_html=True)

if selected == "Comparaison de pays":
    st.markdown('<div class="gradient-text">Comparaison de pays</div>', unsafe_allow_html=True)
    
    # Création des filtres
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pays1 = st.selectbox(
            "Sélectionner le premier pays",
            options=sorted(df['Country'].unique().tolist()),
            index=list(sorted(df['Country'].unique())).index('France') if 'France' in df['Country'].unique() else 0
        )
    
    with col2:
        default_index = list(sorted(df['Country'].unique())).index('United States') if 'United States' in df['Country'].unique() else 0
        pays2 = st.selectbox(
            "Sélectionner le deuxième pays",
            options=sorted(df['Country'].unique().tolist()),
            index=default_index
        )
    
    with col3:
        annee = st.selectbox(
            "Sélectionner l'année",
            options=sorted(df['Year'].unique().tolist()),
            index=len(df['Year'].unique()) - 1
        )

    if pays1 and pays2 and annee:
        # Filtrer les données pour l'année sélectionnée
        df_filtered = df[df['Year'] == annee]
        
        # Obtenir les groupes de variables
        variable_groups = get_variable_groups()
        
        # Créer un onglet pour chaque groupe de variables
        tabs = st.tabs(list(variable_groups.keys()))
        
        for tab, (groupe, group_info) in zip(tabs, variable_groups.items()):
            with tab:
                # Afficher la description du groupe
                st.markdown(f"*{group_info['description']}*")
                
                # Créer et afficher le graphique pour ce groupe
                fig = create_comparison_chart(
                    df_filtered, 
                    pays1, 
                    pays2, 
                    groupe, 
                    group_info['variables']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Afficher le tableau détaillé
                with st.expander("Voir les données détaillées"):
                    data_pays1 = df_filtered[df_filtered['Country'] == pays1].iloc[0]
                    data_pays2 = df_filtered[df_filtered['Country'] == pays2].iloc[0]
                    
                    # Créer un DataFrame pour l'affichage
                    comparison_df = pd.DataFrame({
                        'Variable': group_info['variables'],
                        pays1: [format_value(data_pays1[var], group_info.get('format')) for var in group_info['variables']],
                        pays2: [format_value(data_pays2[var], group_info.get('format')) for var in group_info['variables']],
                        'Différence': [
                            format_value(data_pays1[var] - data_pays2[var], group_info.get('format'))
                            for var in group_info['variables']
                        ]
                    })
                    st.dataframe(comparison_df, use_container_width=True)

elif selected == "Carte":
    create_kepler_map()

elif selected == "Prédictions":
    st.markdown('<div class="gradient-text">Prédictions de l\'espérance de vie</div>', unsafe_allow_html=True)
    
    # Préparation du modèle
    @st.cache_resource
    def get_prediction_model():
        model, features, scaler, feature_importance = prepare_prediction_model(df)
        return model, features, scaler, feature_importance
    
    try:
        model, selected_features, scaler, feature_importance = get_prediction_model()
        
        # Enlever les styles qui causent la bande bleue
        st.markdown("""
            <style>
            div[data-testid="stVerticalBlock"] > div:first-child {
                box-shadow: none !important;
                border: none !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Sélection du pays
        selected_country = st.selectbox(
            "Sélectionner un pays",
            options=sorted(df['Country'].unique().tolist()),
            index=list(sorted(df['Country'].unique())).index('France') if 'France' in df['Country'].unique() else 0
        )
        
        # Sélection de l'année cible
        last_year = df['Year'].max()
        target_year = st.number_input(
            "Année cible pour la prédiction",
            min_value=last_year + 1,
            max_value=last_year + 50,
            value=last_year + 10
        )
        
        years_ahead = target_year - last_year
        
        if st.button("Prédire l'espérance de vie", use_container_width=True):
            country_data = df[df['Country'] == selected_country]
            
            if not country_data.empty:
                # Faire la prédiction
                predictions = predict_life_expectancy(model, selected_features, scaler, country_data, years_ahead)
                final_prediction = predictions[-1]
                current_life_expectancy = country_data[country_data['Year'] == last_year]['Life_expectancy'].iloc[0]
                
                # Afficher le résultat
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        f"Espérance de vie actuelle ({last_year})",
                        f"{current_life_expectancy:.1f} ans"
                    )
                
                with col2:
                    st.metric(
                        f"Espérance de vie prédite ({target_year})",
                        f"{final_prediction:.1f} ans",
                        f"{final_prediction - current_life_expectancy:.1f} ans"
                    )
                
                # Graphique
                fig = go.Figure()
                
                # Données historiques
                historical_data = country_data[['Year', 'Life_expectancy']].sort_values('Year')
                fig.add_trace(go.Scatter(
                    x=[historical_data['Year'].iloc[-1], target_year],
                    y=[current_life_expectancy, final_prediction],
                    mode='lines+markers',
                    name='Projection',
                    line=dict(color='#ff7f0e', dash='dot')
                ))
                
                fig.update_layout(
                    title=f"Projection de l'espérance de vie pour {selected_country}",
                    xaxis_title="Année",
                    yaxis_title="Espérance de vie (années)",
                    template="plotly_white",
                    showlegend=False,
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Note explicative
                st.info(
                    "Cette prédiction est basée sur les tendances historiques et les caractéristiques actuelles du pays. "
                    "De nombreux facteurs externes peuvent influencer l'espérance de vie future."
                )
            
            else:
                st.error("Données non disponibles pour ce pays.")
                
    except Exception as e:
        st.error(f"Une erreur s'est produite : {str(e)}")

elif selected == "Accueil":
    # Titre principal avec animation CSS
    st.markdown("""
        <style>
        .gradient-text {
            background: linear-gradient(45deg, #1e88e5, #00acc1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1em;
            animation: gradient 3s ease infinite;
        }
        
        @keyframes gradient {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        
        .card {
            padding: 1em;
            border-radius: 5px;
            background-color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1em;
        }
        
        .highlight {
            color: #1e88e5;
            font-weight: bold;
        }
        
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
            transition: transform 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        </style>
        
        <div class="gradient-text">
            Analyse Mondiale de l'Espérance de Vie
        </div>
    """, unsafe_allow_html=True)

    # Introduction
    st.markdown("""
        <div class="card">
            Ce dashboard interactif permet d'explorer et d'analyser les facteurs influençant l'espérance de vie à travers le monde. 
            Les données proviennent de l'<span class="highlight">Organisation Mondiale de la Santé (OMS)</span> et ont été enrichies 
            avec des informations de la <span class="highlight">Banque Mondiale</span>.
        </div>
    """, unsafe_allow_html=True)

    # Statistiques rapides
    st.subheader("📊 Statistiques Clés")
    col1, col2, col3, col4 = st.columns(4)

    # Calcul des statistiques
    avg_life_expectancy = df['Life_expectancy'].mean()
    max_life_expectancy = df['Life_expectancy'].max()
    min_life_expectancy = df['Life_expectancy'].min()
    num_countries = df['Country'].nunique()

    with col1:
        st.markdown("""
            <div class="stat-card">
                <h3 style="color: #1e88e5;">Espérance de vie moyenne</h3>
                <h2>{:.1f} ans</h2>
            </div>
        """.format(avg_life_expectancy), unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="stat-card">
                <h3 style="color: #1e88e5;">Maximum</h3>
                <h2>{:.1f} ans</h2>
            </div>
        """.format(max_life_expectancy), unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="stat-card">
                <h3 style="color: #1e88e5;">Minimum</h3>
                <h2>{:.1f} ans</h2>
            </div>
        """.format(min_life_expectancy), unsafe_allow_html=True)

    with col4:
        st.markdown("""
            <div class="stat-card">
                <h3 style="color: #1e88e5;">Pays étudiés</h3>
                <h2>{}</h2>
            </div>
        """.format(num_countries), unsafe_allow_html=True)

    # Section interactive
    st.subheader("🌍 Explorer les Données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_year = st.slider(
            "Sélectionnez une année",
            min_value=int(df['Year'].min()),
            max_value=int(df['Year'].max()),
            value=int(df['Year'].max())
        )

    with col2:
        view_type = st.selectbox(
            "Choisissez un type de visualisation",
            ["Espérance de vie moyenne par région", "Distribution des pays"]
        )

    # Création de la visualisation selon le choix
    df_year = df[df['Year'] == selected_year]

    if view_type == "Espérance de vie moyenne par région":
        fig = px.bar(
            df_year.groupby('Region')['Life_expectancy'].mean().reset_index(),
            x='Region',
            y='Life_expectancy',
            title=f"Espérance de vie moyenne par région en {selected_year}",
            labels={'Life_expectancy': 'Espérance de vie (années)', 'Region': 'Région'},
            color_discrete_sequence=['#1e88e5']
        )
    else:
        fig = px.box(
            df_year,
            x='Region',
            y='Life_expectancy',
            title=f"Distribution de l'espérance de vie par région en {selected_year}",
            labels={'Life_expectancy': 'Espérance de vie (années)', 'Region': 'Région'}
        )

    fig.update_layout(
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # Section "En savoir plus"
    st.subheader("📚 En savoir plus")
    with st.expander("À propos des données"):
        st.markdown("""
            Les données analysées dans ce dashboard proviennent de plusieurs sources fiables :
            
            - **Organisation Mondiale de la Santé (OMS)** : Données sur les vaccinations, l'IMC, le VIH, et les taux de mortalité
            - **Banque Mondiale** : Données sur la population, le PIB et l'espérance de vie
            - **Our World in Data (Université d'Oxford)** : Informations sur la scolarisation
            
            Les données ont été nettoyées et les valeurs manquantes ont été traitées selon deux approches :
            1. Utilisation de la moyenne des trois années les plus proches
            2. Utilisation de la moyenne régionale
            
            Pour garantir la qualité des analyses, les pays avec trop de données manquantes ont été exclus.
        """)

    # Notes de bas de page
    st.markdown("""
        <div style='margin-top: 50px; text-align: center; color: #666;'>
            <p>Projet Data Visualization - ISUP 2024<br>
            Source des données : OMS, Banque Mondiale, Our World in Data, Kaggle</p>
        </div>
    """, unsafe_allow_html=True)

elif selected == "Évolution dans le temps":
    st.markdown('<div class="gradient-text">Évolution dans le temps</div>', unsafe_allow_html=True)
    
    # Ajout du style CSS cohérent avec les autres onglets
    st.markdown("""
        <style>
        div[data-testid="stHorizontalBlock"] {
            background-color: #FFFFFF;
            padding: 15px;
            border: 1px solid #8BA6BC;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Création des colonnes pour les filtres
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Définition des pays par défaut (3 développés, 3 en développement)
        default_countries = ["France", "United States", "Japan", "India", "Brazil", "South Africa"]
        
        # Création de la liste des pays avec l'option "All"
        available_countries = ["All"] + sorted(df['Country'].unique().tolist())
        
        # Vérification que les pays par défaut existent dans le dataset
        valid_default_countries = [country for country in default_countries if country in df['Country'].unique()]
        
        selected_countries = st.multiselect(
            "Sélectionner les pays à comparer",
            options=available_countries,
            default=valid_default_countries
        )
    
    with col2:
        # Sélection de la variable à analyser
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        selected_variable = st.selectbox(
            "Sélectionner la variable à analyser",
            options=numeric_columns,
            index=numeric_columns.index('Life_expectancy') if 'Life_expectancy' in numeric_columns else 0
        )
        
    with col3:
        # Sélection du type de graphique
        chart_type = st.selectbox(
            "Sélectionner le type de graphique",
            options=["Barres", "Lignes", "Points", "Points + Lignes"],
            index=0  # Par défaut sur "Lignes"
        )
    
    # Gestion de la sélection "All"
    if "All" in selected_countries:
        selected_countries = df['Country'].unique().tolist()
    elif not selected_countries:  # Si aucun pays n'est sélectionné
        st.warning("Veuillez sélectionner au moins un pays pour afficher le graphique.")
        selected_countries = []
    
    # Filtrer les données selon les sélections
    if selected_countries:
        filtered_df = df[df['Country'].isin(selected_countries)].copy()
        # Trier le DataFrame par année de manière croissante
        filtered_df = filtered_df.sort_values('Year')
        
        # Utilisation d'une colonne unique pour maintenir la cohérence visuelle
        col1, = st.columns(1)
        with col1:
            st.markdown(f'#### Evolution de {selected_variable} par pays', unsafe_allow_html=True)
            
            # Création du graphique selon le type sélectionné
            if chart_type == "Barres":
                fig = px.bar(
                    filtered_df,
                    x="Country",
                    y=selected_variable,
                    color="Country",
                    animation_frame="Year",
                    animation_group="Country",
                    range_y=[filtered_df[selected_variable].min() * 0.95, filtered_df[selected_variable].max() * 1.05]
                )
            
            elif chart_type == "Lignes":
                animation_speed = st.slider("Vitesse d'animation (ms)", 100, 1000, 300, step=100)
                # Créer une trace pour chaque pays
                fig = px.line(
                    filtered_df,
                    x="Year",
                    y=selected_variable,
                    color="Country",
                    line_shape="linear",
                    markers=False,
                    range_x=[filtered_df['Year'].min(), filtered_df['Year'].max()],
                    range_y=[filtered_df[selected_variable].min() * 0.95,
                            filtered_df[selected_variable].max() * 1.05]
                )
                
                # Ajouter l'animation en modifiant les traces
                for trace in fig.data:
                    trace.update(
                        mode='lines',  # S'assurer que nous sommes en mode 'lines'
                        x=[],  # Commencer avec des listes vides
                        y=[]
                    )
                
                # Créer les frames d'animation
                frames = []
                years = sorted(filtered_df['Year'].unique())
                for year in years:
                    frame_data = []
                    for country in filtered_df['Country'].unique():
                        country_data = filtered_df[
                            (filtered_df['Country'] == country) & 
                            (filtered_df['Year'] <= year)
                        ]
                        frame_data.append(
                            go.Scatter(
                                x=country_data['Year'],
                                y=country_data[selected_variable],
                                mode='lines',
                                name=country
                            )
                        )
                    frames.append(go.Frame(data=frame_data, name=str(year)))
                
                fig.frames = frames
                
                # Configurer l'animation
                fig.update_layout(
                    updatemenus=[{
                        'type': 'buttons',
                        'showactive': False,
                        'buttons': [{
                            'label': 'Play',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': animation_speed, 'redraw': True},
                                'fromcurrent': True,
                                'transition': {'duration': animation_speed//2}
                            }]
                        }]
                    }],
                    # Ajouter un slider pour naviguer dans les années
                    sliders=[{
                        'currentvalue': {'prefix': 'Year: '},
                        'steps': [{'args': [[str(year)]], 'label': str(year), 'method': 'animate'} 
                                for year in years]
                    }]
                )

            elif chart_type == "Points":
                fig = px.scatter(
                    filtered_df,
                    x="Year",
                    y=selected_variable,
                    color="Country",
                    size_max=10
                )
            else:  # "Points + Lignes"
                fig = px.line(
                    filtered_df,
                    x="Year",
                    y=selected_variable,
                    color="Country",
                    line_shape="linear",
                    markers=True
                )
            
            # Personnalisation du graphique
            fig.update_layout(
                xaxis_title="Année" if chart_type != "Barres" else "Pays",
                yaxis_title=selected_variable,
                showlegend=True,
                template="plotly_white"
            )
            
            # Personnalisation de l'animation seulement pour le graphique en barres
            if chart_type == "Barres":
                fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000
                fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 500
            
            # Afficher le graphique
            st.plotly_chart(fig, use_container_width=True)
        
        # Ajouter un tableau de données sous le graphique avec les années dans l'ordre croissant
        with st.expander("Voir les données"):
            pivot_table = filtered_df.pivot_table(
                index='Year',
                columns='Country',
                values=selected_variable
            ).round(2)
            # Trier l'index (années) dans l'ordre croissant
            pivot_table = pivot_table.sort_index()
            st.dataframe(pivot_table)

elif selected == "Analyses":
    st.markdown('<div class="gradient-text">Analyses descriptives</div>', unsafe_allow_html=True)
    
    # Sous-onglets pour différentes analyses
    #analyse_tab = st.radio(
    #    "Choisir une analyse",
    #    ["Histogramme", "Boite à moustaches", "Corrélations"],
    #    horizontal=True
    #)
    
    analyse_tab = st.tabs(["Histogramme", "Boite à moustaches", "Corrélations"])

    with analyse_tab[0]:
        st.subheader("Histogramme")
    
        # Ajout d'options de personnalisation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Sélection de la variable à analyser
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            selected_var = st.selectbox(
                "Sélectionner la variable à analyser",
                options=numeric_columns,
                index=numeric_columns.index('Life_expectancy') if 'Life_expectancy' in numeric_columns else 0
            )
        
        with col2:
            nb_bins = st.slider("Nombre de bins", 10, 100, 30)
        
        with col3:
            color = st.color_picker("Couleur", "#1E88E5")

        # Style CSS
        st.markdown("""
            <style>
            div[data-testid="stHorizontalBlock"] {
                background-color: #FFFFFF;
                padding: 15px;
                border: 1px solid #8BA6BC;
                border-radius: 5px;
            }
            </style>
        """, unsafe_allow_html=True)

        # Graphique dans une colonne
        col1, = st.columns(1)
        with col1:
            st.markdown(f'#### Distribution de {selected_var}', unsafe_allow_html=True)
            fig = px.histogram(
                df,
                x=selected_var,
                template='plotly_dark',
                nbins=nb_bins,
                color_discrete_sequence=[color]
            )
            st.plotly_chart(fig, use_container_width=True)
        
    with analyse_tab[1]:
        st.subheader("Boîte à moustaches")
        st.markdown("""
            <style>
            div[data-testid="stHorizontalBlock"] {
                background-color: #FFFFFF;
                padding: 15px;
                border: 1px solid #8BA6BC;
                border-radius: 5px;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Ajout des filtres
        col1, col2 = st.columns(2)
        
        with col1:
            grouping_var = st.selectbox(
                "Grouper par",
                options=["Status", "Region", "Status et Region"],
                index=0
            )
            
        with col2:
            # Sélection de la variable à analyser
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            selected_var_box = st.selectbox(
                "Variable à analyser",
                options=numeric_columns,
                index=numeric_columns.index('Life_expectancy') if 'Life_expectancy' in numeric_columns else 0,
                key='boxplot_var'  # Clé unique pour éviter le conflit avec le selectbox précédent
            )
        
        col1, = st.columns(1)
        with col1:
            if grouping_var == "Status et Region":
                fig = px.violin(
                    df,
                    x='Region',
                    y=selected_var_box,
                    color='Status',
                    template='plotly_dark',
                    box=True,
                )
            else:
                fig = px.violin(
                    df,
                    x=grouping_var,
                    y=selected_var_box,
                    color=grouping_var,
                    template='plotly_dark',
                    box=True,
                )
            st.plotly_chart(fig, use_container_width=True)
    
    with analyse_tab[2]: #elif analyse_tab == "Corrélations":
        st.subheader("Matrice de corrélation")
        st.markdown("""
            <style>
            div[data-testid="stHorizontalBlock"] {
                background-color: #FFFFFF;
                padding: 15px;
                border: 1px solid #8BA6BC;
                border-radius: 5px;
            }
            </style>
        """, unsafe_allow_html=True)
        # Options de personnalisation
        col1, col2 = st.columns(2)
        with col1:
            color_scale = st.selectbox(
                "Palette de couleurs",
                ["Greens", "Viridis", "RdBu", "Reds", "Blues"],
                index=4
            )
        with col2:
            decimal_places = st.slider(
                "Nombre de décimales",
                0, 4, 2
            )
        
        # Liste des variables
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        columns_list = ["All"] + sorted(numeric_columns)
        
        selected_vars = st.multiselect(
            "Sélectionner les variables pour la matrice de corrélation",
            options=columns_list
        )
        
        if selected_vars:
            if "All" in selected_vars:
                correlation_matrix = df[numeric_columns].corr()
            else:
                correlation_matrix = df[selected_vars].corr()
            
            col1, = st.columns(1)  # Utilisez une seule colonne pour la largeur totale
            with col1:    
                fig = px.imshow(
                    correlation_matrix,
                    color_continuous_scale=color_scale,
                    aspect='auto'
                )
                fig.update_traces(
                    text=correlation_matrix.round(decimal_places), 
                    texttemplate='%{text}'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Veuillez sélectionner au moins une variable pour afficher la matrice de corrélation")

elif selected == "À propos":
    st.markdown('<div class="gradient-text">À propos</div>', unsafe_allow_html=True)
    
    # Création des sous-onglets
    about_tab = st.tabs(["Source des données", "Présentation", "Visualisation"])
    
    # Premier onglet : Lien vers la source des données
    with about_tab[0]:
        st.header("Source des données")
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <a href="https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who" target="_blank" 
               style="background-color: #FF4B4B; 
                      color: white; 
                      padding: 10px 20px; 
                      border-radius: 5px; 
                      text-decoration: none; 
                      font-weight: bold;">
                Accéder aux données originales
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        Ces données proviennent du site Kaggle où elles sont mises à disposition. Pour aux données, cliquez sur le bouton ci-dessus.

        Le jeu de données contenait des données inexactes et beaucoup de valeurs manquantes.
        Le jeu de données est maintenant complètement mis à jour.
        Les données concernant la Population, le PIB et l'Espérance de vie ont 
        été mises à jour selon les données de la Banque Mondiale. Les informations sur 
        les vaccinations contre la Rougeole, l'Hépatite B, la Polio et la Diphtérie, la consommation d'alcool, 
        l'IMC, les cas de VIH, les taux de mortalité et la maigreur ont été collectées à partir des jeux de 
        données publics de l'Organisation Mondiale de la Santé. Les informations sur la Scolarisation ont été collectées 
        auprès de "Our World in Data", un projet de l'Université d'Oxford.

        Le jeu de données présentait quelques valeurs manquantes. Plusieurs **stratégies de remplissage des données 
        manquantes** ont été appliquées :
        1. Remplissage avec la **moyenne des trois années les plus proches**. 
        Si un pays avait une valeur manquante pour une année donnée, la donnée a été complétée avec la moyenne des 
        trois années les plus proches.
        2. Remplissage avec la **moyenne de la Région**. Si un pays avait des valeurs manquantes pour toutes 
        les années, les données ont été complétées avec la moyenne de la Région (par exemple Asie, Afrique, 
        Union Européenne, etc.)

        Les données sont ajustées et les valeurs manquantes sont complétées. Les pays qui avaient plus de 4 c
        olonnes de données manquantes ont été retirés de la base de données. Par exemple le Soudan, l
        e Soudan du Sud et la Corée du Nord.

        La base de données possède une variable qui classe les pays en deux groupes : pays **Développés vs En développement**. 
        Selon l'Organisation Mondiale du Commerce, chaque pays se définit lui-même comme "Développé" ou "En développement". 
        Par conséquent, il est difficile de catégoriser les pays. L'ONU dispose d'une liste datant de 2014 qui, à des fins 
        d'analyse, classe les pays comme économies développées, en transition et en développement. Les pays ayant des 
        économies en transition présentent des caractéristiques similaires aux pays catégorisés comme développés ou en 
        développement. Les pays ont été regroupés selon leur Revenu National Brut par habitant. En conséquence, les nations 
        ont été divisées en quatre groupes de revenus : revenu élevé, revenu intermédiaire supérieur, revenu intermédiaire 
        inférieur et faible revenu. Les niveaux de Revenu Intérieur Brut sont fixés par la Banque Mondiale pour assurer la comparabilité.
                """)
    
    # Deuxième onglet : Présentation des données
    with about_tab[1]:
        st.header("Présentation des données")
        st.write("""
        ### Context
        
        Bien que de nombreuses études aient été menées par le passé sur les facteurs influençant l'espérance de vie, 
        en tenant compte des variables démographiques, de la composition des revenus et des taux de mortalité, il a été constaté que 
        l'effet de la vaccination et de l'indice de développement humain n'avait pas été pris en compte auparavant. De plus, certaines 
        recherches antérieures ont été réalisées en utilisant une régression linéaire multiple basée sur des données d'une seule année 
        pour tous les pays. Par conséquent, cela motive la résolution de ces deux facteurs mentionnés précédemment en formulant un modèle de 
        régression basé sur un modèle à effets mixtes et une régression linéaire multiple, tout en considérant les données sur une période 
        allant de 2000 à 2015 pour tous les pays. Des vaccinations importantes comme l'hépatite B, la polio et la diphtérie seront également 
        prises en compte. En résumé, cette étude se concentrera sur les facteurs de vaccination, les facteurs de mortalité, les facteurs économiques, 
        les facteurs sociaux et d'autres facteurs liés à la santé. Étant donné que les observations de cet ensemble de données sont basées sur 
        différents pays, il sera plus facile pour un pays de déterminer le facteur prédictif qui contribue à une valeur plus faible de l'espérance 
        de vie. Cela aidera à suggérer à un pays quels domaines devraient être privilégiés afin d'améliorer efficacement l'espérance de vie de sa population.
        
        ### Contenu
        Le projet repose sur l'exactitude des données. Le référentiel de données de l'Observatoire mondial de la santé (GHO) sous l'Organisation 
        mondiale de la santé (OMS) suit l'état de santé ainsi que de nombreux autres facteurs connexes pour tous les pays. Les ensembles de données 
        sont mis à la disposition du public à des fins d'analyse des données de santé. L'ensemble de données relatives à l'espérance de vie et aux 
        facteurs de santé pour 193 pays a été collecté sur le même site web du référentiel de données de l'OMS, et les données économiques correspondantes 
        ont été collectées sur le site web des Nations Unies. Parmi toutes les catégories de facteurs liés à la santé, seuls les facteurs critiques 
        les plus représentatifs ont été choisis. Il a été observé qu'au cours des 15 dernières années, il y a eu un développement considérable dans 
        le secteur de la santé, entraînant une amélioration des taux de mortalité humaine, particulièrement dans les pays en développement, par rapport 
        aux 30 années précédentes. Par conséquent, dans ce projet, nous avons considéré les données de 2000 à 2015 pour 193 pays pour une analyse plus 
        approfondie. Les fichiers de données individuels ont été fusionnés en un seul ensemble de données. Une inspection visuelle initiale des données 
        a révélé quelques valeurs manquantes. Comme les ensembles de données provenaient de l'OMS, nous n'avons trouvé aucune erreur évidente. Les données 
        manquantes ont été traitées dans le logiciel R en utilisant la commande Missmap. Le résultat a indiqué que la plupart des données manquantes 
        concernaient la population, l'hépatite B et le PIB. Les données manquantes provenaient de pays moins connus comme le Vanuatu, les Tonga, le Togo, 
        le Cap-Vert, etc. Il était difficile de trouver toutes les données pour ces pays et il a donc été décidé d'exclure ces pays de 
        l'ensemble de données du modèle final. Le fichier fusionné final (ensemble de données final) comprend 22 colonnes et 2938 lignes, 
        ce qui représente 20 variables prédictives. Toutes les variables prédictives ont ensuite été divisées en plusieurs grandes catégories : 
        facteurs liés à la vaccination, facteurs de mortalité, facteurs économiques et facteurs sociaux.
        
        ### Objectifs
        L'ensemble de données vise à répondre aux questions clés suivantes :

        1. Les différents facteurs prédictifs qui ont été choisis initialement affectent-ils réellement 
        l'espérance de vie ? Quelles sont les variables prédictives qui affectent réellement l'espérance de vie ?
        2. Un pays ayant une valeur d'espérance de vie plus faible (<65 ans) devrait-il augmenter ses dépenses de santé afin d'améliorer 
        sa durée de vie moyenne ?
        3. Comment les taux de mortalité infantile et adulte affectent-ils l'espérance de vie ?
        4. L'espérance de vie a-t-elle une corrélation positive ou négative avec les habitudes alimentaires, le mode de vie, l'exercice physique, le tabagisme, la consommation d'alcool, etc. ?
        5. Quel est l'impact de la scolarisation sur la durée de vie des êtres humains ?
        6. L'espérance de vie a-t-elle une relation positive ou négative avec la consommation d'alcool ?
        7. Les pays densément peuplés ont-ils tendance à avoir une espérance de vie plus faible ?
        8. Quel est l'impact de la couverture vaccinale sur l'espérance de vie ?
        """)
        
    # Troisième onglet : Visualisation des données
    with about_tab[2]:
        st.header("Visualisation des données")
        
        # Création d'un sélecteur pour choisir le type de visualisation
        viz_type = st.selectbox(
            "Choisir le type de visualisation",
            ["Table complète", "Statistiques descriptives"]
        )
        
        if viz_type == "Table complète":
            st.dataframe(df)
            
        elif viz_type == "Statistiques descriptives":
            st.write("Statistiques descriptives des données numériques")
            st.dataframe(df.describe())