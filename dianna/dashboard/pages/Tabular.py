import numpy as np
import seaborn as sns
import streamlit as st
from _model_utils import load_data
from _model_utils import load_labels
from _model_utils import load_model
from _model_utils import load_penguins
from _model_utils import load_sunshine
from _model_utils import load_training_data
from _models_tabular import explain_tabular_dispatcher
from _models_tabular import predict
from _shared import _get_top_indices_and_labels
from _shared import _methods_checkboxes
from _shared import add_sidebar_logo
from _shared import reset_example
from _shared import reset_method
from st_aggrid import AgGrid
from st_aggrid import GridOptionsBuilder
from st_aggrid import GridUpdateMode
from dianna.utils.downloader import download
from dianna.visualization import plot_tabular

add_sidebar_logo()

def description_explainer(open='open'):
    """Expandable text section with image."""
    return (st.markdown(
            f"""
            <details {open}>
            <summary><b>Description of the explanation</b></summary>

            The explanation is visualised as a **relevance bar-chart** for the top (up to 10) most
            relevant _attributes (features)_. <br>
            The chart displays the relevance _attributions_ of the individual features of the tabular data
            to a **pretrained model**'s classification or regression prediciton.
            The attribution chart can be computed for any predicted outcome.

            The attribution colormap
            assigns :blue[**blue**] color to negative relevances,
            and :red[**red**] color to positive values.
            </details>
            """,
            unsafe_allow_html=True
           ),
           st.text("")
           )

st.title('Explaining Tabular data classification/regression')

st.sidebar.header('Input data')

input_type = st.sidebar.radio(
        label='Select which input to use',
        options = ('Use an example', 'Use your own data'),
        index = None,
        on_change = reset_example,
        key = 'Tabular_input_type'
    )

# Use the examples
if input_type == 'Use an example':
    load_example = st.sidebar.radio(
        label='Use example',
        options=('Sunshine hours prediction (regression)', 'Penguin identification (classification)'),
        index = None,
        on_change = reset_method,
        key='Tabular_load_example')

    if load_example == "Sunshine hours prediction (regression)":
        tabular_data_file = download('weather_prediction_dataset_light.csv', 'data')
        tabular_model_file = download('sunshine_hours_regression_model.onnx', 'model')
        tabular_training_data_file = tabular_data_file
        tabular_label_file = None

        training_data, data = load_sunshine(tabular_data_file)
        labels =  None

        mode = 'regression'
        description_explainer("")
        st.markdown(
        """
        *****************************************************************************
        This example demonstrates the use of DIANNA on a pre-trained [regression
        model](https://zenodo.org/records/10580833) to predict tomorrow's sunshine hours
        based on meteorological data from today.
        The model is trained on the
        [weather prediction dataset](https://zenodo.org/records/5071376). <br>
        The meteorological data includes measurements (features) of
        _cloud coverage, humidity, air pressure, global radiation, precipitation_, and
        _mean, min_ and _max temeprature_
        for various European cities.
        """,
        unsafe_allow_html=True )

    elif load_example == 'Penguin identification (classification)':
        tabular_model_file = download('penguin_model.onnx', 'model')
        data_penguins = sns.load_dataset('penguins')
        labels = data_penguins['species'].unique()

        training_data, data = load_penguins(data_penguins)

        mode = 'classification'
        description_explainer("")
        st.markdown(
        """
        ****************************************************************************
        This example demonstrates the use of DIANNA on a pre-trained [classification
        model](https://zenodo.org/records/10580743) to identify if a penguin belongs to one of three different species
        based on a number of measurable physical characteristics. <br>
        The model is trained on the
        [penguin dataset](https://www.kaggle.com/code/parulpandey/penguin-dataset-the-new-iris).
        The penguin characteristics include the _bill length_, _bill depth_, _flipper length_, and _body mass_.
        """,
        unsafe_allow_html=True)
    else:
        description_explainer()
        st.info('Select an example in the left panel to coninue')
        st.stop()

# Option to upload your own data
if input_type == 'Use your own data':
    tabular_data_file = st.sidebar.file_uploader('Select tabular data', type='csv')
    tabular_model_file = st.sidebar.file_uploader('Select model', type='onnx')
    tabular_training_data_file = st.sidebar.file_uploader('Select training data', type='npy')
    tabular_label_file = st.sidebar.file_uploader('Select labels in case of classification model', type='txt')

    if not (tabular_data_file and tabular_model_file and tabular_training_data_file):
        description_explainer()
        st.info('Add your input data in the left panel to continue')
        st.stop()
    else:
        description_explainer("")

    data = load_data(tabular_data_file)
    model = load_model(tabular_model_file)
    training_data = load_training_data(tabular_training_data_file)

    if tabular_label_file:
        labels = load_labels(tabular_label_file)
        mode = 'classification'
    else:
        labels = None
        mode = 'regression'

if input_type is None:
    description_explainer()
    st.info('Select which input type to use in the left panel to continue')
    st.stop()

model = load_model(tabular_model_file)
serialized_model = model.SerializeToString()

choices = ('RISE', 'LIME', 'KernelSHAP')

st.text("")

# Get predictions and create parameter box
with st.container(border=True):
    prediction_placeholder = st.empty()
    methods, method_params = _methods_checkboxes(choices=choices, key='Tabular_cb')


# Configure Ag-Grid options
gb = GridOptionsBuilder.from_dataframe(data)
gb.configure_selection('single')
grid_options = gb.build()

# Display the grid with the DataFrame
grid_response = AgGrid(
    data,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    theme='streamlit'
)

if grid_response['selected_rows'] is not None:
    selected_row = int(grid_response['selected_rows'].index[0])
    selected_data = data.iloc[selected_row].to_numpy()[1:]
    with st.spinner('Predicting class'):
        predictions = predict(model=serialized_model, tabular_input=selected_data.reshape(1,-1))

    with prediction_placeholder:
        top_indices, top_labels = _get_top_indices_and_labels(
            predictions=predictions[0], labels=labels)

else:
    st.info("Select the input data by clicking a row in the table.")
    st.stop()

st.text("")
st.text("")

weight = 0.85 / len(methods)
column_spec = [0.15, *[weight for _ in methods]]

_, *columns = st.columns(column_spec)
for col, method in zip(columns, methods):
    col.markdown(f"<h4 style='text-align: center; '>{method}</h4>", unsafe_allow_html=True)

for index, label in zip(top_indices, top_labels):
    index_col, *columns = st.columns(column_spec)

    if mode == 'classification':
        index_col.markdown(f'##### Class: {label}')

    for col, method in zip(columns, methods):
        kwargs = method_params[method].copy()
        kwargs['mode'] = mode
        kwargs['_feature_names']=data.columns.to_list()[1:]

        func = explain_tabular_dispatcher[method]

        with col:
            with st.spinner(f'Running {method}'):
                relevances = func(serialized_model, selected_data, training_data, **kwargs)
            if mode == 'classification':
                plot_relevances = relevances[np.argmax(predictions)]
            else:
                plot_relevances = relevances

            fig, _ = plot_tabular(x=plot_relevances, y=kwargs['_feature_names'],
                                  num_features=10, show_plot=False)
            st.pyplot(fig)

    # add some white space to separate rows
    st.markdown('')


st.stop()
