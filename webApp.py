import joblib
import pandas as pd
import ast
import string
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import nltk
import re

# Download the required NLTK datasets
nltk.download('stopwords')
nltk.download('punkt')

# Initialize the app with a different Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])

# Load the data
url = 'https://raw.githubusercontent.com/cosylabiiit/SustainableFoodDB/main/Data/Recipe_cfp.csv'
df = pd.read_csv(url, header=None, names=['RecipeID', 'Ingredients', 'Region'])

# Function to safely parse ingredients
def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []

# Process the data
df['Ingredients'] = df['Ingredients'].apply(safe_literal_eval)
regions = [region for region in df['Region'].unique() if region.lower() != 'region']

# Load the pre-trained model, vectorizer, and cuisine map
rf_model = joblib.load('rf_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
cuisine_map = joblib.load('cuisine_map.pkl')

# Define stopwords
stopwords_list = nltk.corpus.stopwords.words('english') + list(string.punctuation)
stopwords_list += ["''", '""', '...', '``', 'tsp', 'tbsp', 'tablespoon', 'teaspoon', 'tablespoons', 'teaspoons',
                   'large', 'cup', 'ounces', 'pound', 'oz', 'slice', 'sliced', 'cup', 'cups', 'ounce', 'ounces',
                   'chopped', 'finely', 'cut', 'thinly', 'pounds', 'lb', 'lbs', 'g', 'oz', 'small', 'large']

# Function to process ingredients
def process_ingredients(ingredients):
    tokens = nltk.word_tokenize(ingredients)
    stopwords_removed = ' '.join([token.lower() for token in tokens if token not in stopwords_list])
    pattern = r"[a-z]+"
    regex_tokens = re.findall(pattern, stopwords_removed)
    return ' '.join(regex_tokens)

# Define the layout of the app
app.layout = dbc.Container(
    [
        dbc.NavbarSimple(
            brand="Sustainable Food Recipes",
            brand_href="#",
            color="primary",
            dark=True,
            className="mb-4"
        ),
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.H4("Predict Cuisine", className="mb-4"),
                        dcc.Input(
                            id='ingredients-input',
                            type='text',
                            placeholder='Enter ingredients separated by commas',
                            className='mb-4',
                            style={'width': '100%'}
                        ),
                        html.Button('Predict', id='predict-button', className='btn btn-primary mb-4'),
                        html.Div(id='prediction-result', className='mt-4')
                    ],
                    className='p-4 bg-light rounded shadow-sm'
                ),
                width=12,
                className='mb-4'
            )
        ),
        dbc.Row(
            dbc.Col(
                dcc.Dropdown(
                    id='region-dropdown',
                    options=[{'label': region, 'value': region} for region in regions],
                    value=regions[0],  # Default value excluding 'Region'
                    placeholder="Select a region",
                    className="mb-4"
                ),
                width=6
            )
        ),
        dbc.Row(
            dbc.Col(dcc.Graph(id='ingredient-graph'), width=12)
        ),
        dbc.Row(
            dbc.Col(dcc.Graph(id='recipe-count-graph'), width=12, className="mt-4")
        ),
        dbc.Row(
            dbc.Col(dcc.Graph(id='avg-ingredients-graph'), width=12, className="mt-4")
        ),
        dbc.Row(
            dbc.Col(html.Footer("Footer content here", className="text-center mt-4"), width=12)
        )
    ],
    fluid=True,
)

def predict_cuisine(n_clicks, ingredients):
    if n_clicks is None:
        return ''
    if not ingredients:
        return 'Please enter some ingredients.'

    # Process the ingredients input
    ingredients_processed = process_ingredients(ingredients)
    ingredients_vectorized = vectorizer.transform([ingredients_processed])

    # Predict the cuisine
    prediction = rf_model.predict(ingredients_vectorized)
    prediction_proba = rf_model.predict_proba(ingredients_vectorized)

    # Map prediction back to cuisine
    cuisine_map_inverse = {v: k for k, v in cuisine_map.items()}
    predicted_cuisine = cuisine_map_inverse[prediction[0]]
    confidence = max(prediction_proba[0]) * 100

    return f'Predicted Cuisine: {predicted_cuisine} with {confidence:.2f}% confidence.'

# Define the callback to update the graphs based on the selected region
@app.callback(
    [Output('ingredient-graph', 'figure'),
     Output('recipe-count-graph', 'figure'),
     Output('avg-ingredients-graph', 'figure')],
    [Input('region-dropdown', 'value')]
)
def update_graphs(selected_region):
    filtered_df = df[df['Region'] == selected_region]

    # Ingredient Graph
    all_ingredients = filtered_df['Ingredients'].explode()
    ingredient_counts = all_ingredients.value_counts().head(10)  # Top 10 ingredients
    ingredient_figure = {
        'data': [{'x': ingredient_counts.index, 'y': ingredient_counts.values, 'type': 'bar', 'name': 'Ingredients'}],
        'layout': {'title': f'Top 10 Ingredients in {selected_region}', 'plot_bgcolor': '#f9f9f9',
                   'paper_bgcolor': '#f9f9f9'}
    }

    # Recipe Count per Region
    recipe_count = df['Region'].value_counts()
    recipe_count_figure = {
        'data': [{'x': recipe_count.index, 'y': recipe_count.values, 'type': 'bar', 'name': 'Recipe Count'}],
        'layout': {'title': 'Recipe Count per Region', 'plot_bgcolor': '#f9f9f9', 'paper_bgcolor': '#f9f9f9'}
    }

    # Average Number of Ingredients per Region
    df['NumIngredients'] = df['Ingredients'].apply(len)
    avg_ingredients = df.groupby('Region')['NumIngredients'].mean().sort_values()
    avg_ingredients_figure = {
        'data': [{'x': avg_ingredients.index, 'y': avg_ingredients.values, 'type': 'bar', 'name': 'Avg Ingredients'}],
        'layout': {'title': 'Average Number of Ingredients per Region', 'plot_bgcolor': '#f9f9f9',
                   'paper_bgcolor': '#f9f9f9'}
    }

    return ingredient_figure, recipe_count_figure, avg_ingredients_figure

# Define the callback to predict the cuisine
@app.callback(
    Output('prediction-result', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('ingredients-input', 'value')]
)
def update_output(n_clicks, value):
    return predict_cuisine(n_clicks, value)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
