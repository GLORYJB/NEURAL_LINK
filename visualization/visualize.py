import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy import signal



def interactive_plot(data, plot_type='raw'):
    """
    Create interactive plots for neural data visualization.
    """
    if plot_type == 'raw':
        return create_raw_plot(data)
    elif plot_type == 'frequency':
        return create_frequency_plot(data)
    elif plot_type == 'wavelet':
        return create_wavelet_plot(data)
    else:
        return create_statistical_plot(data)


def create_raw_plot(data):
    """
    Create raw signal visualization with annotations.
    """
    fig = go.Figure()

    for column in data.columns:
        fig.add_trace(go.Scatter(
            y=data[column],
            name=column,
            mode='lines',
            line=dict(width=1),
            hovertemplate="Value: %{y:.2f}<br>Time: %{x}<extra></extra>"
        ))

    fig.update_layout(
        template="plotly_white",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode='x unified'
    )

    return fig


def create_frequency_plot(data):
    """
    Create frequency domain visualization.
    """
    fig = go.Figure()

    for column in data.columns:
        freqs, psd = signal.welch(data[column])
        fig.add_trace(go.Scatter(
            x=freqs,
            y=psd,
            name=column,
            mode='lines',
            line=dict(width=1),
            hovertemplate="Frequency: %{x:.2f}<br>Power: %{y:.2e}<extra></extra>"
        ))

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power Spectral Density",
        showlegend=True,
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode='x unified'
    )

    return fig


def create_wavelet_plot(data):
    """
    Create wavelet transform visualization.
    """
    fig = go.Figure()


    fig.update_layout(
        template="plotly_white",
        xaxis_title="Time",
        yaxis_title="Frequency",
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode='closest'
    )

    return fig


def create_statistical_plot(data):
    """
    Create statistical visualization with box plots and violin plots.
    """
    fig = go.Figure()

    for column in data.columns:
        fig.add_trace(go.Box(
            y=data[column],
            name=column,
            boxpoints='outliers',
            jitter=0.3,
            pointpos=-1.8
        ))

        fig.add_trace(go.Violin(
            y=data[column],
            name=column,
            side='positive',
            line_color='lightgray',
            fillcolor='rgba(0,0,0,0.1)',
            showlegend=False
        ))

    fig.update_layout(
        template="plotly_white",
        boxmode='group',
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode='closest'
    )

    return fig


def create_correlation_plot(data):
    """
    Create correlation matrix visualization.
    """
    corr_matrix = data.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        hovertemplate="Channel 1: %{x}<br>Channel 2: %{y}<br>Correlation: %{z:.2f}<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40),
        height=600
    )

    return fig