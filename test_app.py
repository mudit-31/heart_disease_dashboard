import dash
from dash import html

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Test Dash App"),
    html.P("If this appears, Dash is working!")
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)