# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
# Dash&Flask
from jupyter_dash import JupyterDash
import dash
from dash import html
from dash.exceptions import PreventUpdate
from flask_caching import Cache
# Onnx
import onnx
from onnx_tf.backend import prepare
# Others
from PIL import Image
from html2image import Html2Image
import dianna
from dianna.utils.tokenizers import SpacyTokenizer
import spacy
import os
import base64
import layouts
import utilities
from utilities import MovieReviewsModelRunner, _create_html
import numpy as np
import warnings
warnings.filterwarnings('ignore')  # disable warnings relateds to versions of tf

folder_on_server = "app_data"
os.makedirs(folder_on_server, exist_ok=True)
tokenizer = SpacyTokenizer()  # for now always use SpacyTokenizer, needs to be changed

# Build App
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = JupyterDash(
    __name__,
    external_stylesheets=external_stylesheets,
    prevent_initial_callbacks=True,
    suppress_callback_exceptions=True)

# Caching
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache'
})
cache.clear()

# global variables
class_name_mnist = ['digit 0', 'digit 1']
class_name_text = ["negative", "positive"]

try:
    spacy.load("en_core_web_sm")
except Exception:  # If not present, we download
    spacy.cli.download("en_core_web_sm")
    spacy.load("en_core_web_sm")

# ########################## Images page ###########################


# uploading test image
@app.callback(dash.dependencies.Output('graph_test', 'figure'),
              dash.dependencies.Input('upload-image', 'contents'),
              dash.dependencies.State('upload-image', 'filename'))
def upload_image(contents, filename):
    """Takes in test image file, returns it as a Plotly figure."""
    if contents is not None:

        try:

            if 'jpg' in filename[0]:

                _, content_string = contents[0].split(',')

                with open(os.path.join(folder_on_server, filename[0]), 'wb') as f:
                    f.write(base64.b64decode(content_string))

                data_path = os.path.join(folder_on_server, filename[0])

                X_test = utilities.open_image(data_path)

                fig = go.Figure()

                if X_test.shape[2] < 3:  # it's grayscale

                    fig.add_trace(go.Heatmap(z=X_test[:, :, 0], colorscale='gray', showscale=False))

                fig.update_layout(
                    width=300,
                    height=300,
                    title=f"{filename[0]} uploaded",
                    title_x=0.5,
                    title_font_color=layouts.colors['blue1'])

                fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
                fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

                fig.layout.paper_bgcolor = layouts.colors['blue4']

                return fig

            return utilities.blank_fig(
                text='File format error! <br><br>Please upload only images in .jpg format.')

        except Exception as e:
            print(e)
            return utilities.blank_fig(
                    text='There was an error processing this file.')
    else:
        return utilities.blank_fig()


# uploading model for image
@app.callback(dash.dependencies.Output('output-model-img-upload', 'children'),
              dash.dependencies.Input('upload-model-img', 'contents'),
              dash.dependencies.State('upload-model-img', 'filename'))
def upload_model_img(contents, filename):
    """Takes in the model file, returns a print statement about its uploading state."""
    if contents is not None:
        try:
            if 'onnx' in filename[0]:

                _, content_string = contents[0].split(',')

                with open(os.path.join(folder_on_server, filename[0]), 'wb') as f:
                    f.write(base64.b64decode(content_string))

                return html.Div([f'{filename[0]} uploaded'])

            return html.Div([
                html.P('File format error!'),
                html.Br(),
                html.P('Please upload only models in .onnx format.')
                ])
        except Exception as e:
            print(e)
            return html.Div(['There was an error processing this file.'])
    else:
        raise PreventUpdate


# perform expensive computations in this "global store"
# these computations are cached in a globally available
# redis memory store which is available across processes
# and for all time.
@cache.memoize()
def global_store_i(method_sel, model_path, image_test):
    """Takes in the selected XAI method, the model path and the image to test, returns the explainations array."""
    # expensive query
    if method_sel == "RISE":
        relevances = dianna.explain_image(
            model_path, image_test, method=method_sel,
            labels=list(range(2)),
            n_masks=5000, feature_res=8, p_keep=.1,
            axis_labels=('height', 'width', 'channels'))

    elif method_sel == "KernelSHAP":
        relevances = dianna.explain_image(
            model_path, image_test,
            method=method_sel, nsamples=1000,
            background=0, n_segments=200, sigma=0,
            axis_labels=('height', 'width', 'channels'))

    else:
        relevances = dianna.explain_image(
            model_path, image_test * 256, 'LIME',
            axis_labels=('height', 'width', 'channels'),
            random_state=2,
            labels=list(range(2)),
            preprocess_function=utilities.preprocess_function)

    return relevances


# signaling
@app.callback(
    dash.dependencies.Output('signal_image', 'data'),
    [dash.dependencies.Input('method_sel_img', 'value'),
     dash.dependencies.State("upload-model-img", "filename"),
     dash.dependencies.State("upload-image", "filename"),
     ])
def compute_value_i(method_sel, fn_m, fn_i):
    """Takes in the selected XAI method, the model and the image filenames, returns the selected XAI method."""
    if (method_sel is None) or (fn_m is None) or (fn_i is None):
        raise PreventUpdate

    for m in method_sel:
        # compute value and send a signal when done
        data_path = os.path.join(folder_on_server, fn_i[0])
        image_test = utilities.open_image(data_path)

        model_path = os.path.join(folder_on_server, fn_m[0])

        try:
            global_store_i(m, model_path, image_test)
        except Exception:
            return method_sel

    return method_sel


# update image explainations
@app.callback(
    dash.dependencies.Output('output-state-img', 'children'),
    dash.dependencies.Output('graph_img', 'figure'),
    dash.dependencies.State("upload-model-img", "filename"),
    dash.dependencies.State("upload-image", "filename"),
    dash.dependencies.Input("signal_image", "data"),
    dash.dependencies.Input("upload-model-img", "filename"),
    dash.dependencies.Input("upload-image", "filename"),
)
# pylint: disable=too-many-locals
# pylint: disable=unused-argument
def update_multi_options_i(fn_m, fn_i, sel_methods, new_model, new_image):
    """Takes in the last model and image uploaded filenames, the selected XAI method, and returns the selected XAI method."""
    ctx = dash.callback_context

    if (ctx.triggered[0]["prop_id"] == "upload-model-img.filename") or (ctx.triggered[0]["prop_id"] == "upload-image.filename") or (not ctx.triggered):
        cache.clear()
        return html.Div(['']), utilities.blank_fig()
    if (not sel_methods):
        return html.Div(['']), utilities.blank_fig()

    # update graph
    # pylint: disable=too-many-nested-blocks
    if (fn_m and fn_i) is not None:

        data_path = os.path.join(folder_on_server, fn_i[0])
        X_test = utilities.open_image(data_path)

        onnx_model_path = os.path.join(folder_on_server, fn_m[0])
        onnx_model = onnx.load(onnx_model_path)
        # get the output node
        output_node = prepare(onnx_model, gen_tensor_dict=True).outputs[0]

        try:
            predictions = prepare(onnx_model).run(X_test[None, ...])[f'{output_node}']

            if len(predictions[0]) == 2:
                class_name = class_name_mnist

            pred_class = class_name[np.argmax(predictions)]

            n_rows = len(class_name)

            fig = make_subplots(rows=n_rows, cols=3, subplot_titles=("RISE", "KernelShap", "LIME"))  # , horizontal_spacing = 0.05)

            for m in sel_methods:

                for i in range(n_rows):

                    fig.update_yaxes(title_text=class_name[i], row=i+1, col=1)

                    if m == "RISE":

                        try:
                            relevances_rise = global_store_i(
                                m, onnx_model_path, X_test)

                            # RISE plot
                            fig.add_trace(go.Heatmap(
                                                z=X_test[:, :, 0], colorscale='gray', showscale=False), i+1, 1)
                            fig.add_trace(go.Heatmap(
                                                z=relevances_rise[i], colorscale='Bluered',
                                                showscale=False, opacity=0.7), i+1, 1)

                        except Exception:
                            return html.Div(['There was an error running the model. Check either the test image or the model.']), utilities.blank_fig()

                    elif m == "KernelSHAP":

                        shap_values, segments_slic = global_store_i(
                            m, onnx_model_path, X_test)

                        # KernelSHAP plot
                        fig.add_trace(go.Heatmap(
                                        z=X_test[:, :, 0], colorscale='gray', showscale=False), i+1, 2)
                        fig.add_trace(go.Heatmap(
                                        z=utilities.fill_segmentation(shap_values[i][0], segments_slic),
                                        colorscale='Bluered',
                                        showscale=False,
                                        opacity=0.7), i+1, 2)
                    else:

                        relevances_lime = global_store_i(
                            m, onnx_model_path, X_test)

                        # LIME plot
                        fig.add_trace(go.Heatmap(
                                            z=X_test[:, :, 0], colorscale='gray', showscale=False), i+1, 3)

                        fig.add_trace(go.Heatmap(
                                            z=relevances_lime[i], colorscale='Bluered',
                                            showscale=False, opacity=0.7), i+1, 3)

            fig.update_layout(
                width=650,
                height=500,
                paper_bgcolor=layouts.colors['blue4'])

            fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
            fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

            return html.Div(['The predicted class is: ' + pred_class], style={
                'fontSize': 14,
                'margin-top': '20px',
                'margin-right': '40px'
                }), fig

        except Exception as e:
            print(e)
            return html.Div(['There was an error running the model. Check either the test image or the model.']), utilities.blank_fig()
    else:
        return html.Div(['Missing either model or image.']), utilities.blank_fig()

###################################################################

# ########################## Text page ###########################


# uploading test text
@app.callback(dash.dependencies.Output('text_test', 'children'),
              dash.dependencies.Input('submit-text', 'n_clicks'),
              dash.dependencies.State('upload-text', 'value'))
def upload_text(clicks, input_value):
    """Takes in test text string, and print it on the dashboard."""
    if clicks is not None:
        return html.Div([
                    html.P('Input string for the model is:'),
                    html.Br(),
                    html.P(f'"{input_value}"')
                    ])

    return html.Div(['No string uploaded.'])


# uploading model for the text
@app.callback(dash.dependencies.Output('output-model-text-upload', 'children'),
              dash.dependencies.Input('upload-model-text', 'contents'),
              dash.dependencies.State('upload-model-text', 'filename'))
def upload_model_text(contents, filename):
    """Takes in the model file, returns a print statement about its uploading state."""
    if contents is not None:
        try:
            if 'onnx' in filename[0]:

                _, content_string = contents[0].split(',')

                with open(os.path.join(folder_on_server, filename[0]), 'wb') as f:
                    f.write(base64.b64decode(content_string))

                return html.Div([f'{filename[0]} uploaded'])

            return html.Div([
                html.P('File format error!'),
                html.Br(),
                html.P('Please upload only models in .onnx format.')
                ])

        except Exception as e:
            print(e)
            return html.Div(['There was an error processing this file.'])
    else:
        raise PreventUpdate


# perform expensive computations in this "global store"
# these computations are cached in a globally available
# redis memory store which is available across processes
# and for all time.
@cache.memoize()
def global_store_t(method_sel, model_runner, input_text):
    """Takes in the selected XAI method, the model path and the string to test, returns the explainations highlighted on the string itself."""
    predictions = model_runner(input_text)
    class_name = class_name_text
    pred_class = class_name[np.argmax(predictions)]
    labels = tuple(class_name_text)
    pred_idx = labels.index(pred_class)


    # expensive query
    relevances = dianna.explain_text(
        model_runner,
        input_text,
        tokenizer,
        method_sel,
        labels=[pred_idx]
        )

    return relevances


# signaling
@app.callback(
    dash.dependencies.Output('signal_text', 'data'),
    [dash.dependencies.Input('method_sel_text', 'value'),
     dash.dependencies.State("upload-model-text", "filename"),
     dash.dependencies.State("upload-text", "value"),
     ])
def compute_value_t(method_sel, fn_m, input_text):
    """Takes in the selected XAI method, the model filename and the text, returns the selected XAI method."""
    if (method_sel is None) or (fn_m is None) or (input_text is None):
        raise PreventUpdate

    word_vector_path = '../tutorials/data/movie_reviews_word_vectors.txt'
    model_path = os.path.join(folder_on_server, fn_m[0])
    model_runner = MovieReviewsModelRunner(model_path, word_vector_path, max_filter_size=5)

    for m in method_sel:
        # compute value and send a signal when done
        try:
            global_store_t(m, model_runner, input_text)
        except Exception:
            return method_sel
    return method_sel


# update text explainations
@app.callback(
    dash.dependencies.Output("output-state-text", "children"),
    dash.dependencies.Output("graph_text_lime", "figure"),
    dash.dependencies.Output("graph_text_rise", "figure"),
    dash.dependencies.State("upload-model-text", "filename"),
    dash.dependencies.State("upload-text", "value"),
    dash.dependencies.Input("signal_text", "data"),
    dash.dependencies.Input("upload-model-text", "filename"),
    dash.dependencies.Input("upload-text", "value"),
)
# pylint: disable=too-many-locals
# pylint: disable=unused-argument
def update_multi_options_t(fn_m, input_text, sel_methods, new_model, new_text):
    """Takes in the last model filename and text uploaded, the selected XAI method, and returns the selected XAI method."""
    ctx = dash.callback_context

    if (ctx.triggered[0]["prop_id"] == "upload-model-text.filename") or (ctx.triggered[0]["prop_id"] == "upload-text.value") or (not ctx.triggered):
        cache.clear()
        return html.Div(['']), utilities.blank_fig(), utilities.blank_fig()
    if (not sel_methods):
        return html.Div(['']), utilities.blank_fig(), utilities.blank_fig()

    # update text explainations
    if (fn_m and input_text) is not None:

        word_vector_path = '../tutorials/data/movie_reviews_word_vectors.txt'
        onnx_model_path = os.path.join(folder_on_server, fn_m[0])

        print(onnx_model_path)

        # define model runner. max_filter_size is a property of the model
        model_runner = MovieReviewsModelRunner(onnx_model_path, word_vector_path, max_filter_size=5)

        try:
            input_tokens = tokenizer.tokenize(input_text)
            predictions = model_runner(input_text)
            class_name = class_name_text
            pred_class = class_name[np.argmax(predictions)]

            fig_l = utilities.blank_fig()
            fig_r = utilities.blank_fig()

            for m in sel_methods:
                if m == "LIME":

                    relevances_lime = global_store_t(
                        m, model_runner, input_text)

                    output = _create_html(input_tokens, relevances_lime[0], max_opacity=0.8)
                    hti = Html2Image()
                    expl_path = 'text_expl.jpg'

                    hti.screenshot(output, save_as=expl_path)

                    im = Image.open(expl_path)
                    im = np.asarray(im).astype(np.float32)

                    fig_l = px.imshow(im)
                    fig_l.update_xaxes(showgrid=False, range=[0, 1000], showticklabels=False, zeroline=False)
                    fig_l.update_yaxes(showgrid=False, range=[200, 0], showticklabels=False, zeroline=False)
                    fig_l.update_layout(
                        title='LIME explaination:',
                        title_font_color=layouts.colors['blue1'],
                        paper_bgcolor=layouts.colors['blue4'],
                        plot_bgcolor=layouts.colors['blue4'],
                        height=200,
                        width=500,
                        margin_b=40,
                        margin_t=40,
                        margin_l=0,
                        margin_r=0
                        )

                elif m == "RISE":

                    relevances_rise = global_store_t(
                        m, model_runner, input_text)

                    output = _create_html(input_tokens, relevances_rise[0], max_opacity=0.8)
                    hti = Html2Image()
                    expl_path = 'text_expl.jpg'

                    hti.screenshot(output, save_as=expl_path)

                    im = Image.open(expl_path)
                    im = np.asarray(im).astype(np.float32)

                    fig_r = px.imshow(im)
                    fig_r.update_xaxes(showgrid=False, range=[0, 1000], showticklabels=False, zeroline=False)
                    fig_r.update_yaxes(showgrid=False, range=[200, 0], showticklabels=False, zeroline=False)
                    fig_r.update_layout(
                        title='RISE explaination:',
                        title_font_color=layouts.colors['blue1'],
                        paper_bgcolor=layouts.colors['blue4'],
                        plot_bgcolor=layouts.colors['blue4'],
                        height=200,
                        width=500,
                        margin_b=10,
                        margin_t=40,
                        margin_l=0,
                        margin_r=0)

            return html.Div(['The predicted class is: ' + pred_class]), fig_l, fig_r

        except Exception:
            return html.Div([
                'There was an error running the model. Check either the test text or the model.'
                ]), utilities.blank_fig(), utilities.blank_fig()
    else:
        return html.Div(['Missing either model or input text.']), utilities.blank_fig(), utilities.blank_fig()

###################################################################
