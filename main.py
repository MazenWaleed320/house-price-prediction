import math

import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import dash
from dash import dcc, html, Input, Output, State

# ==========================
# Load Data
# ==========================
data = pd.read_csv("Housing.csv")


# Features
num_features = ["area", "bedrooms", "bathrooms", "stories", "parking"]
cat_features = ["mainroad", "guestroom", "basement",
                "hotwaterheating", "airconditioning",
                "prefarea", "furnishingstatus"]



for feature in num_features:
    f = pd.Series(data[feature])
    Q1 = f.quantile(0.25)
    Q3 = f.quantile(0.75)
    IQR = Q3 - Q1
    data = data[data[feature] < Q3 + 1.1 * IQR]
    data = data[data[feature] > Q1 - 1.1  * IQR]



price = pd.Series(data['price'])
pQ1 = price.quantile(0.25)
pQ3 = price.quantile(0.75)
pIQR = pQ3 - pQ1
data = data[data['price'] < pQ3 + 1.1 * pIQR]
data = data[data['price'] > pQ1 - 1.1 * pIQR]

data['sq_area'] = data['area'] ** 0.5
num_features[0] = 'sq_area'

X = data.drop("price", axis=1)
y = data["price"]

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

# Pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit model
model.fit(X_train, y_train)

# ==========================
# Model Evaluation
# ==========================
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


eval_fig = px.scatter(
    x=y_test, y=y_pred,
    labels={'x': 'Actual Price', 'y': 'Predicted Price'},
    title="Predicted vs Actual Prices"
)
eval_fig.add_shape(
    type="line",
    x0=y_test.min(), y0=y_test.min(),
    x1=y_test.max(), y1=y_test.max(),
    line=dict(color="red", dash="dash")
)

# ==========================
# Dash App
# ==========================
app = dash.Dash(__name__)

# Figures for Numerical Features
fig_area = px.scatter(data, x="area", y="price", trendline="ols", title="Price vs Area")
fig_parking = px.bar(data.groupby("parking")["price"].mean().reset_index(),
                     x="parking", y="price", title="Average Price by Parking")
fig_stories = px.bar(data.groupby("stories")["price"].mean().reset_index(),
                     x="stories", y="price", title="Average Price by Stories")
fig_bedrooms = px.bar(data.groupby("bedrooms")["price"].mean().reset_index(),
                      x="bedrooms", y="price", title="Average Price by Bedrooms")
fig_bathrooms = px.bar(data.groupby("bathrooms")["price"].mean().reset_index(),
                       x="bathrooms", y="price", title="Average Price by Bathrooms")

# ==========================
# Layout
# ==========================
app.layout = html.Div([
    html.H1("🏠 Housing Price Dashboard", style={"textAlign": "center"}),

    dcc.Tabs([
        dcc.Tab(label="Numerical Analysis", children=[
            dcc.Graph(figure=fig_area),
            dcc.Graph(figure=fig_parking),
            dcc.Graph(figure=fig_stories),
            dcc.Graph(figure=fig_bedrooms),
            dcc.Graph(figure=fig_bathrooms),
        ]),

        dcc.Tab(label="Categorical Analysis", children=[
            *[dcc.Graph(
                figure=px.box(data, x=col, y="price", title=f"Price by {col}")
            ) for col in ["mainroad", "guestroom", "basement",
                          "hotwaterheating", "airconditioning", "prefarea"]],
            dcc.Graph(
                figure=px.bar(
                    data.groupby("furnishingstatus")["price"].mean().reset_index(),
                    x="furnishingstatus", y="price",
                    title="Average Price by Furnishing Status"
                )
            )
        ]),

        dcc.Tab(label="Model Evaluation & Prediction", children=[
            html.H3("📊 Model Performance"),
            html.P(f"MSE : {mse:,.0f}"),
            html.P(f"RMSE: {rmse:,.0f}"),
            html.P(f"R² Score: {r2:.3f}"),
            dcc.Graph(figure=eval_fig),

            html.H3("🔮 Predict House Price"),
            html.Div([
                html.Label("Area:"),
                dcc.Input(id="area", type="number", value=2000),
                html.Label("Bedrooms:"),
                dcc.Input(id="bedrooms", type="number", value=3),
                html.Label("Bathrooms:"),
                dcc.Input(id="bathrooms", type="number", value=2),
                html.Label("Stories:"),
                dcc.Input(id="stories", type="number", value=2),
                html.Label("Parking:"),
                dcc.Input(id="parking", type="number", value=1),

                html.Label("Mainroad:"),
                dcc.Dropdown(id="mainroad", options=[{"label": v, "value": v} for v in data["mainroad"].unique()],
                             value="yes"),
                html.Label("Guestroom:"),
                dcc.Dropdown(id="guestroom", options=[{"label": v, "value": v} for v in data["guestroom"].unique()],
                             value="no"),
                html.Label("Basement:"),
                dcc.Dropdown(id="basement", options=[{"label": v, "value": v} for v in data["basement"].unique()],
                             value="no"),
                html.Label("Hotwaterheating:"),
                dcc.Dropdown(id="hotwaterheating", options=[{"label": v, "value": v} for v in data["hotwaterheating"].unique()],
                             value="no"),
                html.Label("Airconditioning:"),
                dcc.Dropdown(id="airconditioning", options=[{"label": v, "value": v} for v in data["airconditioning"].unique()],
                             value="yes"),
                html.Label("Prefarea:"),
                dcc.Dropdown(id="prefarea", options=[{"label": v, "value": v} for v in data["prefarea"].unique()],
                             value="yes"),
                html.Label("Furnishing Status:"),
                dcc.Dropdown(id="furnishingstatus", options=[{"label": v, "value": v} for v in data["furnishingstatus"].unique()],
                             value="furnished"),

                html.Br(),
                html.Button("Predict", id="predict-btn", n_clicks=0),
                html.Div(id="prediction-output", style={"marginTop": "20px", "fontWeight": "bold", "fontSize": "20px"})
            ])
        ])
    ])
])

# ==========================
# Callbacks
# ==========================
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("area", "value"),
    State("bedrooms", "value"),
    State("bathrooms", "value"),
    State("stories", "value"),
    State("parking", "value"),
    State("mainroad", "value"),
    State("guestroom", "value"),
    State("basement", "value"),
    State("hotwaterheating", "value"),
    State("airconditioning", "value"),
    State("prefarea", "value"),
    State("furnishingstatus", "value")
)
def predict_price(n_clicks, area, bedrooms, bathrooms, stories, parking,
                  mainroad, guestroom, basement, hotwaterheating,
                  airconditioning, prefarea, furnishingstatus):
    if n_clicks > 0:
        input_data = pd.DataFrame([{
            "sq_area": area ** 0.5,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "parking": parking,
            "mainroad": mainroad,
            "guestroom": guestroom,
            "basement": basement,
            "hotwaterheating": hotwaterheating,
            "airconditioning": airconditioning,
            "prefarea": prefarea,
            "furnishingstatus": furnishingstatus
        }])
        pred = model.predict(input_data)[0]
        return f"🏠 Predicted House Price: {pred:,.0f}"
    return ""

# ==========================
# Run App
# ==========================
if __name__ == "__main__":
    app.run(debug=True)
