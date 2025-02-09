import dash
import numpy as np
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash_iconify import DashIconify
from processing.preprocess import load_data, preprocess_data
from visualization.visualize import interactive_plot

# Load and preprocess data
file_path = "data/simulated_neural_data.csv"
data = load_data(file_path)
processed_data = preprocess_data(data)

app = dash.Dash(
    __name__,
    external_stylesheets=[
        'https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css'
    ]
)


sidebar = html.Div([
    html.Div([

        html.Div([
            DashIconify(icon="carbon:brain", width=32, className="text-blue-500"),
            html.H2("NeuroViz Pro", className="text-xl font-bold text-blue-600")
        ], className="flex items-center space-x-3 mb-6"),

        # Main Controls
        html.Div([

            html.Div([
                html.Label("Channels", className="font-medium text-gray-700"),
                dcc.Dropdown(
                    id="channel-dropdown",
                    options=[{"label": f"Channel {col}", "value": col} for col in processed_data.columns],
                    value=[processed_data.columns[0]],
                    multi=True,
                    className="mb-4"
                )
            ], className="mb-6"),


            html.Div([
                html.Label("Filters", className="font-medium text-gray-700"),
                dcc.Checklist(
                    id="signal-filters",
                    options=[
                        {'label': 'Notch Filter (60Hz)', 'value': 'notch'},
                        {'label': 'Bandpass (1-100Hz)', 'value': 'bandpass'},
                        {'label': 'Artifact Removal', 'value': 'artifact'}
                    ],
                    value=[],
                    className="space-y-2"
                )
            ], className="mb-6"),


            html.Div([
                html.Label("Time Range", className="font-medium text-gray-700"),
                html.Div([
                    html.Button("1m", id="range-1m", className="px-3 py-1 bg-gray-200 rounded-lg"),
                    html.Button("5m", id="range-5m", className="px-3 py-1 bg-gray-200 rounded-lg"),
                    html.Button("15m", id="range-15m", className="px-3 py-1 bg-gray-200 rounded-lg"),
                    html.Button("All", id="range-all", className="px-3 py-1 bg-gray-200 rounded-lg"),
                ], className="flex space-x-2 mb-2"),
                dcc.RangeSlider(
                    id='time-range-slider',
                    min=0,
                    max=len(processed_data) - 1,
                    value=[0, len(processed_data) - 1],
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], className="mb-6"),


            html.Div([
                html.Label("Analysis Settings", className="font-medium text-gray-700"),
                dcc.Tabs(id="analysis-tabs", value='basic', children=[
                    dcc.Tab(label='Basic', value='basic'),
                    dcc.Tab(label='Advanced', value='advanced'),
                    dcc.Tab(label='Custom', value='custom'),
                ], className="mb-4"),
                html.Div(id='analysis-content')
            ], className="mb-6"),
        ]),


        html.Div([
            html.Button([
                DashIconify(icon="carbon:download", className="mr-2"),
                "Export Data"
            ], id="export-button", className="w-full bg-blue-500 text-white px-4 py-2 rounded-lg mb-2"),
            html.Button([
                DashIconify(icon="carbon:document-pdf", className="mr-2"),
                "Generate Report"
            ], id="report-button", className="w-full bg-green-500 text-white px-4 py-2 rounded-lg")
        ], className="mt-auto")
    ], className="h-full flex flex-col p-6")
], className="bg-white w-72 fixed left-0 top-0 h-screen shadow-lg overflow-y-auto")


main_content = html.Div([

    html.Div([
        html.Div([
            html.H1("Neural Signal Analysis", className="text-3xl font-bold"),
            html.P("Real-time signal processing and visualization", className="text-gray-600")
        ]),
        html.Div([
            html.Button([
                DashIconify(icon="carbon:notification", className="mr-2"),
                "Alerts"
            ], className="bg-gray-100 p-2 rounded-lg"),
            html.Button([
                DashIconify(icon="carbon:settings", className="mr-2"),
                "Settings"
            ], className="bg-gray-100 p-2 rounded-lg")
        ], className="flex space-x-3")
    ], className="flex justify-between items-center mb-6"),


    html.Div([
        html.Div([
            DashIconify(icon="carbon:activity", width=24, className="text-blue-500"),
            html.Div([
                html.H3("Signal Quality", className="text-gray-600"),
                html.Div(id="signal-quality", className="text-2xl font-bold")
            ])
        ], className="bg-white p-4 rounded-lg shadow-sm"),
        html.Div([
            DashIconify(icon="carbon:wave-2", width=24, className="text-green-500"),
            html.Div([
                html.H3("Peak Frequency", className="text-gray-600"),
                html.Div(id="peak-frequency", className="text-2xl font-bold")
            ])
        ], className="bg-white p-4 rounded-lg shadow-sm"),
        html.Div([
            DashIconify(icon="carbon:warning", width=24, className="text-yellow-500"),
            html.Div([
                html.H3("Anomalies", className="text-gray-600"),
                html.Div(id="anomaly-count", className="text-2xl font-bold")
            ])
        ], className="bg-white p-4 rounded-lg shadow-sm"),
        html.Div([
            DashIconify(icon="carbon:chart-maximum", width=24, className="text-purple-500"),
            html.Div([
                html.H3("Signal Power", className="text-gray-600"),
                html.Div(id="signal-power", className="text-2xl font-bold")
            ])
        ], className="bg-white p-4 rounded-lg shadow-sm")
    ], className="grid grid-cols-4 gap-4 mb-6"),


    html.Div([
        # Raw Signal
        html.Div([
            html.Div([
                html.H3("Raw Signal", className="text-lg font-bold"),
                html.Button([
                    DashIconify(icon="carbon:maximize", className="w-4 h-4")
                ], className="p-1 hover:bg-gray-100 rounded")
            ], className="flex justify-between items-center mb-2"),
            dcc.Graph(id="neural-plot", className="h-64")
        ], className="bg-white p-4 rounded-lg shadow-sm"),


        html.Div([
            html.Div([
                html.H3("Frequency Spectrum", className="text-lg font-bold"),
                html.Button([
                    DashIconify(icon="carbon:maximize", className="w-4 h-4")
                ], className="p-1 hover:bg-gray-100 rounded")
            ], className="flex justify-between items-center mb-2"),
            dcc.Graph(id="frequency-plot", className="h-64")
        ], className="bg-white p-4 rounded-lg shadow-sm"),


        html.Div([
            html.Div([
                html.H3("Wavelet Transform", className="text-lg font-bold"),
                html.Button([
                    DashIconify(icon="carbon:maximize", className="w-4 h-4")
                ], className="p-1 hover:bg-gray-100 rounded")
            ], className="flex justify-between items-center mb-2"),
            dcc.Graph(id="wavelet-plot", className="h-64")
        ], className="bg-white p-4 rounded-lg shadow-sm"),


        html.Div([
            html.Div([
                html.H3("Channel Correlation", className="text-lg font-bold"),
                html.Button([
                    DashIconify(icon="carbon:maximize", className="w-4 h-4")
                ], className="p-1 hover:bg-gray-100 rounded")
            ], className="flex justify-between items-center mb-2"),
            dcc.Graph(id="correlation-plot", className="h-64")
        ], className="bg-white p-4 rounded-lg shadow-sm"),


        html.Div([
            html.Div([
                html.H3("Statistical Analysis", className="text-lg font-bold"),
                html.Button([
                    DashIconify(icon="carbon:maximize", className="w-4 h-4")
                ], className="p-1 hover:bg-gray-100 rounded")
            ], className="flex justify-between items-center mb-2"),
            dcc.Graph(id="stats-plot", className="h-64")
        ], className="bg-white p-4 rounded-lg shadow-sm"),


        html.Div([
            html.Div([
                html.H3("Event Detection", className="text-lg font-bold"),
                html.Button([
                    DashIconify(icon="carbon:maximize", className="w-4 h-4")
                ], className="p-1 hover:bg-gray-100 rounded")
            ], className="flex justify-between items-center mb-2"),
            dcc.Graph(id="event-plot", className="h-64")
        ], className="bg-white p-4 rounded-lg shadow-sm")
    ], className="grid grid-cols-3 gap-4 mb-6"),


    html.Div([

        html.Div([
            html.H3("Detected Events", className="text-lg font-bold mb-3"),
            dash_table.DataTable(
                id='events-table',
                columns=[
                    {'name': 'Time', 'id': 'time'},
                    {'name': 'Type', 'id': 'type'},
                    {'name': 'Channel', 'id': 'channel'},
                    {'name': 'Magnitude', 'id': 'magnitude'}
                ],
                style_table={'height': '300px', 'overflowY': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '12px'},
                style_header={
                    'backgroundColor': 'rgb(240, 242, 245)',
                    'fontWeight': 'bold'
                }
            )
        ], className="bg-white p-4 rounded-lg shadow-sm"),

        # Statistics Summary
        html.Div([
            html.H3("Channel Statistics", className="text-lg font-bold mb-3"),
            html.Div(id="statistics-content", className="grid grid-cols-2 gap-4")
        ], className="bg-white p-4 rounded-lg shadow-sm")
    ], className="grid grid-cols-2 gap-4")

], className="ml-72 p-8 bg-gray-50 min-h-screen")

# Complete app layout
app.layout = html.Div([sidebar, main_content])



@app.callback(
    [Output("signal-quality", "children"),
     Output("peak-frequency", "children"),
     Output("anomaly-count", "children"),
     Output("signal-power", "children")],
    [Input("channel-dropdown", "value"),
     Input("time-range-slider", "value")]
)
def update_kpis(selected_channels, time_range):

    start, end = time_range
    filtered_data = processed_data.iloc[start:end + 1]

    signal_quality = calculate_signal_quality(filtered_data, selected_channels)
    peak_frequency = calculate_peak_frequency(filtered_data, selected_channels)
    anomaly_count = calculate_anomalies(filtered_data, selected_channels)
    signal_power = calculate_signal_power(filtered_data, selected_channels)

    return signal_quality, peak_frequency, anomaly_count, signal_power



@app.callback(
    [Output("neural-plot", "figure"),
     Output("frequency-plot", "figure"),
     Output("wavelet-plot", "figure"),
     Output("correlation-plot", "figure"),
     Output("stats-plot", "figure"),
     Output("event-plot", "figure")],
    [Input("channel-dropdown", "value"),
     Input("time-range-slider", "value")]
)
def update_charts(selected_channels, time_range):
    start, end = time_range
    filtered_data = processed_data.iloc[start:end + 1]

    # Raw signal plot
    neural_fig = interactive_plot(filtered_data[selected_channels])


    frequency_fig = plot_frequency_spectrum(filtered_data[selected_channels])


    wavelet_fig = plot_wavelet_analysis(filtered_data[selected_channels])


    correlation_fig = plot_channel_correlation(filtered_data[selected_channels])


    stats_fig = plot_statistical_analysis(filtered_data[selected_channels])


    event_fig = plot_event_detection(filtered_data[selected_channels])

    return neural_fig, frequency_fig, wavelet_fig, correlation_fig, stats_fig, event_fig



@app.callback(
    Output("events-table", "data"),
    [Input("channel-dropdown", "value"),
     Input("time-range-slider", "value")]
)
def update_events_table(selected_channels, time_range):
    start, end = time_range
    filtered_data = processed_data.iloc[start:end + 1]


    detected_events = detect_events(filtered_data[selected_channels])
    return detected_events



@app.callback(
    Output("statistics-content", "children"),
    [Input("channel-dropdown", "value"),
     Input("time-range-slider", "value")]
)
def update_statistics(selected_channels, time_range):
    start, end = time_range
    filtered_data = processed_data.iloc[start:end + 1]

    # Example statistics
    statistics = calculate_basic_statistics(filtered_data[selected_channels])

    stats_layout = [
        html.Div([
            html.H4(f"{stat}: {value:.2f}", className="font-semibold text-lg")
        ]) for stat, value in statistics.items()
    ]

    return stats_layout



@app.callback(
    Output("export-button", "n_clicks"),
    [Input("export-button", "n_clicks")],
    prevent_initial_call=True
)
def export_data(n_clicks):
    if n_clicks > 0:

        export_file_path = "data/processed_neural_data.csv"
        processed_data.to_csv(export_file_path)
        return n_clicks


# Callback for generating reports
@app.callback(
    Output("report-button", "n_clicks"),
    [Input("report-button", "n_clicks")],
    prevent_initial_call=True
)
def generate_report(n_clicks):
    if n_clicks > 0:

        generate_pdf_report(processed_data)
        return n_clicks



def calculate_signal_quality(data, selected_channels):

    return np.random.random() * 100  # Simulated value


def calculate_peak_frequency(data, selected_channels):

    return np.random.random() * 50  # Simulated value


def calculate_anomalies(data, selected_channels):

    return np.random.randint(0, 10)  # Simulated value


def calculate_signal_power(data, selected_channels):

    return np.random.random() * 10  # Simulated value


def plot_frequency_spectrum(data):
    # Example frequency spectrum plot using FFT
    freq = np.fft.fftfreq(len(data), 1)
    power = np.abs(np.fft.fft(data)) ** 2
    fig = go.Figure(data=go.Scatter(x=freq, y=power))
    return fig


def plot_wavelet_analysis(data):

    fig = go.Figure(data=go.Scatter(x=np.arange(len(data)), y=data))
    return fig


def plot_channel_correlation(data):

    corr_matrix = np.corrcoef(data.T)
    fig = px.imshow(corr_matrix, text_auto=True)
    return fig


def plot_statistical_analysis(data):

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    fig = go.Figure(data=[go.Bar(x=np.arange(len(mean)), y=mean, name='Mean'),
                          go.Bar(x=np.arange(len(std)), y=std, name='Std Dev')])
    return fig


def plot_event_detection(data):

    fig = go.Figure(data=go.Scatter(x=np.arange(len(data)), y=data))
    return fig


def detect_events(data):

    return [{"time": t, "type": "Event", "channel": "C1", "magnitude": np.random.random()} for t in range(len(data))]


def calculate_basic_statistics(data):

    return {
        "Mean": np.mean(data),
        "Std Dev": np.std(data),
        "Max": np.max(data),
        "Min": np.min(data)
    }


def generate_pdf_report(data):

    pass


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
