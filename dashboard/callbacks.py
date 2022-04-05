# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
import dianna
from dianna import visualization
from dianna import utils
from scipy.special import expit as sigmoid
import spacy
from torchtext.data import get_tokenizer
from torchtext.vocab import Vectors
import os
import base64
import layouts
import utilities
import numpy as np
import dianna
import warnings
warnings.filterwarnings('ignore') # disable warnings relateds to versions of tf

folder_on_server = "app_data"
os.makedirs(folder_on_server, exist_ok=True)

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
class_name_text = ("negative", "positive")

try:
    spacy.load("en_core_web_sm")
except: # If not present, we download
    spacy.cli.download("en_core_web_sm")
    spacy.load("en_core_web_sm")

@app.callback(dash.dependencies.Output('output-model-img-upload', 'children'),
              dash.dependencies.Input('upload-model-img', 'contents'),
              dash.dependencies.State('upload-model-img', 'filename'))
def upload_model(contents, filename):
    if contents is not None:
        try:
            if 'onnx' in filename[0]:

                content_type, content_string = contents[0].split(',')

                with open(os.path.join(folder_on_server, filename[0]), 'wb') as f:
                    f.write(base64.b64decode(content_string))

                return html.Div([f'{filename[0]} uploaded'])
            else:
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

@app.callback(dash.dependencies.Output('output-model-text-upload', 'children'),
              dash.dependencies.Input('upload-model-text', 'contents'),
              dash.dependencies.State('upload-model-text', 'filename'))
def upload_model(contents, filename):
    if contents is not None:
        try:
            if 'onnx' in filename[0]:

                content_type, content_string = contents[0].split(',')

                with open(os.path.join(folder_on_server, filename[0]), 'wb') as f:
                    f.write(base64.b64decode(content_string))

                return html.Div([f'{filename[0]} uploaded'])
            else:
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

@app.callback(dash.dependencies.Output('graph_test', 'figure'),
              dash.dependencies.Input('upload-image', 'contents'),
              dash.dependencies.State('upload-image', 'filename'))
def upload_image(contents, filename):
    if contents is not None:

        try:

            if 'jpg' in filename[0]:

                content_type, content_string = contents[0].split(',')

                with open(os.path.join(folder_on_server, filename[0]), 'wb') as f:
                    f.write(base64.b64decode(content_string))

                data_path = os.path.join(folder_on_server, filename[0])

                X_test = utilities.open_image(data_path)

                fig = go.Figure()

                if X_test.shape[2] < 3: # it's grayscale

                    fig.add_trace(go.Heatmap(z=X_test[:,:,0], colorscale='gray', showscale=False))

                fig.update_layout(
                    width=300,
                    height=300,
                    title=f"{filename[0]} uploaded",
                    title_x=0.5,
                    title_font_color=layouts.colors['blue1'])

                fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
                fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)

                fig.layout.paper_bgcolor = layouts.colors['blue4']

                return fig
            else:
                return utilities.blank_fig(
                    text='File format error! <br><br>Please upload only images in .jpg format.')
        
        except Exception as e:
            print(e)
            return utilities.blank_fig(
                    text='There was an error processing this file.')


@app.callback(dash.dependencies.Output('text_test', 'children'),
              dash.dependencies.Input('submit-text', 'n_clicks'),
              dash.dependencies.State('upload-text', 'value'))
def upload_text(clicks, input_value):
    if clicks is not None:
        return html.Div([
                    html.P('Input string for the model is:'),
                    html.Br(),
                    html.P(f'"{input_value}"')
                    ])
    else:
        return html.Div([f'No string uploaded.'])


# perform expensive computations in this "global store"
# these computations are cached in a globally available
# redis memory store which is available across processes
# and for all time.
@cache.memoize()
def global_store_i(method_sel, model_path, image_test):
    # expensive query
    if method_sel == "RISE":
        relevances = dianna.explain_image(
            model_path, image_test, method=method_sel,
            labels=[i for i in range(2)],
            n_masks=5000, feature_res=8, p_keep=.1,
            axis_labels=('height','width','channels'))

    elif method_sel == "KernelSHAP":
        relevances = dianna.explain_image(
            model_path, image_test,
            method=method_sel, nsamples=1000,
            background=0, n_segments=200, sigma=0,
            axis_labels=('height','width','channels'))

    else:
        relevances = dianna.explain_image(
            model_path, image_test * 256, 'LIME',
            axis_labels=('height','width','channels'),
            random_state=2,
            labels=[i for i in range(2)],
            preprocess_function=utilities.preprocess_function)

    return relevances

@app.callback(
    dash.dependencies.Output('signal', 'data'),
    [dash.dependencies.Input('method_sel', 'value'),
    dash.dependencies.State("upload-model-img", "filename"),
    dash.dependencies.State("upload-image", "filename"),
    ])
def compute_value_i(method_sel, fn_m, fn_i):

    if method_sel is None:
        raise PreventUpdate
    else:
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

@app.callback(
    dash.dependencies.Output('output-state-img', 'children'),
    dash.dependencies.Output('graph', 'figure'),
    dash.dependencies.Input("signal", "data"),
    dash.dependencies.State("upload-model-img", "filename"),
    dash.dependencies.State("upload-image", "filename"),
    dash.dependencies.Input("upload-model-img", "filename"),
    dash.dependencies.Input("upload-image", "filename"),
)
def update_multi_options_i(sel_methods, fn_m, fn_i, new_model, new_image):

    ctx = dash.callback_context

    if (ctx.triggered[0]["prop_id"] == "upload-model-img.filename") or (ctx.triggered[0]["prop_id"] == "upload-image.filename") or (not ctx.triggered):
        cache.clear()
        return html.Div(['']), utilities.blank_fig()
    elif (not sel_methods):
        return html.Div(['']), utilities.blank_fig()
    else:
        # update graph
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
                    class_name = [c for c in class_name_mnist]

                pred_class = class_name[np.argmax(predictions)]

                n_rows = len(class_name)

                fig = make_subplots(rows=n_rows, cols=3, subplot_titles=("RISE", "KernelShap", "LIME"))#, horizontal_spacing = 0.05)

                for m in sel_methods:

                    for i in range(n_rows):

                        fig.update_yaxes(title_text=class_name[i], row=i+1, col=1)

                        if m == "RISE":
                            
                            try:
                                relevances_rise = global_store_i(
                                    m, onnx_model_path, X_test)

                                # RISE plot
                                fig.add_trace(go.Heatmap(
                                                    z=X_test[:,:,0], colorscale='gray', showscale=False), i+1, 1)
                                fig.add_trace(go.Heatmap(
                                                    z=relevances_rise[i], colorscale='Bluered',
                                                    showscale=False, opacity=0.7), i+1, 1)

                            except Exception:
                                html.Div(['There was an error running the model. Check either the test image or the model.']), utilities.blank_fig()

                        elif m == "KernelSHAP":

                            shap_values, segments_slic = global_store_i(
                                m, onnx_model_path, X_test)

                            # KernelSHAP plot
                            fig.add_trace(go.Heatmap(
                                            z=X_test[:,:,0], colorscale='gray', showscale=False), i+1, 2)
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
                                                z=X_test[:,:,0], colorscale='gray', showscale=False), i+1, 3)

                            fig.add_trace(go.Heatmap(
                                                z=relevances_lime[i], colorscale='Bluered',
                                                showscale=False, opacity=0.7), i+1, 3)

                fig.update_layout(
                    width=650,
                    height=500,
                    paper_bgcolor=layouts.colors['blue4'])

                fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
                fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)

                return html.Div(['The predicted class is: ' + pred_class], style = {
                    'fontSize': 14,
                    'margin-top': '20px',
                    'margin-right': '40px'
                    }), fig

            except Exception as e:
                print(e)
                return html.Div(['There was an error running the model. Check either the test image or the model.']), utilities.blank_fig()
        else:
            return html.Div(['Missing either model or image.']), utilities.blank_fig()

@app.callback(
    dash.dependencies.Output('output-state-text', 'children'),
    #dash.dependencies.Input("signal", "data"),
    dash.dependencies.Input("method_sel", "value"),
    dash.dependencies.State("upload-model-text", "filename"),
    dash.dependencies.State("upload-text", "value"),
    dash.dependencies.Input("upload-model-text", "filename"),
    dash.dependencies.Input("upload-text", "value"),
)
def update_multi_options_t(sel_methods, fn_m, input_text, new_model, new_text):

    ctx = dash.callback_context

    if (ctx.triggered[0]["prop_id"] == "upload-model-text.filename") or (ctx.triggered[0]["prop_id"] == "upload-text.value") or (not ctx.triggered):
        cache.clear()
        return html.Div([''])
    elif (not sel_methods):
        return html.Div([''])
    else:
        # update text explainations
        if (fn_m and input_text) is not None:

            word_vector_path = '../tutorials/data/movie_reviews_word_vectors.txt'
            onnx_model_path = os.path.join(folder_on_server, fn_m[0])

            # define model runner. max_filter_size is a property of the model
            model_runner = MovieReviewsModelRunner(onnx_model_path, word_vector_path, max_filter_size=5)

            for m in sel_methods:
                if m=="LIME":
                    # An explanation is returned for each label, but we ask for just one label so the output is a list of length one.
                    relevances = dianna.explain_text(model_runner, input_text, m, label=class_name_text.index('positive'))[0]
                    print(relevances)
                    return html.Div([relevances])
        else:
            return html.Div(['Missing either model or input text.'])


class MovieReviewsModelRunner:
    def __init__(self, model, word_vectors, max_filter_size):
        self.run_model = utils.get_function(model)
        self.vocab = Vectors(word_vectors, cache=os.path.dirname(word_vectors))
        self.max_filter_size = max_filter_size    
        self.tokenizer = get_tokenizer('spacy', 'en_core_web_sm')

    def __call__(self, sentences):
        # ensure the input has a batch axis
        if isinstance(sentences, str):
            sentences = [sentences]

        tokenized_sentences = []
        for sentence in sentences:
            # tokenize and pad to minimum length
            tokens = self.tokenizer(sentence)
            if len(tokens) < self.max_filter_size:
                tokens += ['<pad>'] * (self.max_filter_size - len(tokens))
            
            # numericalize the tokens
            tokens_numerical = [self.vocab.stoi[token] if token in self.vocab.stoi else self.vocab.stoi['<unk>']
                                for token in tokens]
            tokenized_sentences.append(tokens_numerical)
            
        # run the model, applying a sigmoid because the model outputs logits
        logits = self.run_model(tokenized_sentences)
        pred = np.apply_along_axis(sigmoid, 1, logits)
        
        # output two classes
        positivity = pred[:, 0]
        negativity = 1 - positivity
        return np.transpose([negativity, positivity])