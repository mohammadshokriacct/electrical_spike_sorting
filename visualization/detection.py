import logging
import numpy as np
from scipy.optimize import curve_fit

from plotly.subplots import make_subplots
import plotly.graph_objects as go


def activation_curve_plt(amp_list, detection_sig, activation_curve=None, image_path=None, sigmoid_threshold=0.50):
    fig = make_subplots(rows=1, cols=1)

    curve_threshold = None
    try:
        popt, _ = curve_fit(sigmoid, amp_list, detection_sig)
        poly_trend = sigmoid(amp_list, *popt)
        curve_threshold = sigmoid_inv(sigmoid_threshold, *popt)
        curve_threshold = None if ((curve_threshold > np.max(amp_list)) or (curve_threshold < np.min(amp_list))) else curve_threshold
    except:
        deg = 5
        z = np.polyfit(amp_list, detection_sig, deg, rcond=None, full=False, w=None, cov=False)
        p = np.poly1d(z)
        poly_trend = p(np.reshape(amp_list, [-1]))

    fig.add_trace(
        go.Scatter(
            name='Algorithm',
            x=amp_list,
            y=detection_sig,
            mode='markers', 
            marker=dict(
                color="blue"
            )
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            name="Algorithm's Curve",
            x=amp_list,
            y=poly_trend,
            marker=dict(
                color="green"
            )
        ),
        row=1, col=1
    )
    if curve_threshold is not None:
        fig.add_trace(
        go.Scatter(
            x=[curve_threshold, curve_threshold],
            y=[0, sigmoid_threshold],
            mode='lines',
            line = dict(color='green', dash = 'dash'),
            showlegend=False
        ),
        row=1, col=1
        )
        fig.add_hline(y=sigmoid_threshold, line_width=3, line_dash="dash", line_color="LightBlue")

    # Human activation curve
    if activation_curve is not None:
        curve_threshold = None
        try:
            popt, _ = curve_fit(sigmoid, amp_list, activation_curve)
            poly_trend = sigmoid(amp_list, *popt)
            curve_threshold = sigmoid_inv(sigmoid_threshold, *popt)
            curve_threshold = None if ((curve_threshold > np.max(amp_list)) or (curve_threshold < np.min(amp_list))) else curve_threshold
        except:
            deg = 5
            z = np.polyfit(amp_list, activation_curve, deg, rcond=None, full=False, w=None, cov=False)
            p = np.poly1d(z)
            poly_trend = p(np.reshape(amp_list, [-1]))

        fig.add_trace(
            go.Scatter(
                name='Human',
                x=np.reshape(amp_list, [-1]),
                y=activation_curve,
                mode='markers', 
                marker=dict(
                    color="red"
                )
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                name="Human's Curve",
                x=amp_list,
                y=poly_trend,
                marker=dict(
                    color="orange"
                )
            ),
            row=1, col=1
        )
        
        if curve_threshold is not None:
            fig.add_trace(
            go.Scatter(
                name=None,
                x=[curve_threshold, curve_threshold],
                y=[0, sigmoid_threshold],
                mode='lines',
                line = dict(color='orange', dash = 'dash'),
                showlegend=False
            ),
            row=1, col=1
            )

    fig.update_xaxes(range=[0,np.max(amp_list)], showline=True, linewidth=2, linecolor='black', gridcolor='LightGray', griddash='dash', mirror=True)
    fig.update_yaxes(range=[-0.02, 1.05], showline=True, linewidth=2, linecolor='black', gridcolor='LightGray', griddash='dash', mirror=True, \
                    title="Probability")

    fig['layout']['xaxis']['title'] = r"$\text{Amplitude (}\mu\text{A)}$"

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(
        autosize=False,
        showlegend=True,
        margin = {'l':0,'r':0,'t':0,'b':0}, 
        width=800, height=300, font_size=20, font_family="Times New Roman"
    )

    fig.show()
    if image_path is not None:
        logging.info(image_path)
        fig.write_image(image_path)

def sigmoid(x, x0, k):
    y = 1 / (1 + np.exp(-k*(x-x0)))
    return y

def sigmoid_inv(y, x0, k):
    x = x0 - np.log(1/y - 1)/k
    return x