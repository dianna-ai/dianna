import numpy as np
import streamlit as st
from _model_utils import load_data
from _model_utils import load_labels
from _model_utils import load_model
from _model_utils import load_training_data
from _models_tabular import predict
from _models_tabular import explain_tabular_dispatcher
from _shared import _get_top_indices_and_labels
from _shared import _methods_checkboxes
from _shared import add_sidebar_logo
from _shared import reset_example
from _shared import reset_method
from dianna.utils.downloader import download
from dianna.utils.onnx_runner import SimpleModelRunner
from dianna.visualization import plot_tabular
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

add_sidebar_logo()

st.title('Tabular data explanation')

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
    """load_example = st.sidebar.radio(
        label='Use example',
        options=(''),
        index = None,
        on_change = reset_method,
        key='Tabular_load_example')"""
    st.info("No examples availble yet")
    st.stop()

# Option to upload your own data
if input_type == 'Use your own data':
    tabular_data_file = st.sidebar.file_uploader('Select tabular data', type='csv')
    tabular_model_file = st.sidebar.file_uploader('Select model', type='onnx')
    tabular_training_data_file = st.sidebar.file_uploader('Select training data', type='npy')
    tabular_label_file = st.sidebar.file_uploader('Select labels in case of classification model', type='txt')

if input_type is None:
    st.info('Select which input type to use in the left panel to continue')
    st.stop()

if not (tabular_data_file and tabular_model_file and tabular_training_data_file):
    st.info('Add your input data in the left panel to continue')
    st.stop()

data = load_data(tabular_data_file)

model = load_model(tabular_model_file)
serialized_model = model.SerializeToString()

training_data = load_training_data(tabular_training_data_file)

if tabular_label_file:
    labels = load_labels(tabular_label_file)
    mode = 'classification'
else:
    labels = None
    mode = 'regression'

choices = ('RISE', 'LIME', 'KernelSHAP')

st.text("")
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
    selected_row = grid_response['selected_rows']['Index'].iloc[0]
    selected_data = data.iloc[selected_row, 1:].to_numpy(dtype=np.float32)
    with st.spinner('Predicting class'):
        predictions = predict(model=serialized_model, tabular_input=selected_data)

    with prediction_placeholder:
        top_indices, top_labels = _get_top_indices_and_labels(
            predictions=predictions[0], labels=labels)

else:
    st.info("Select the input data either by clicking the corresponding row in the table or input the row index above to continue.")
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

    if mode == 'classification': index_col.markdown(f'##### Class: {label}')

    for col, method in zip(columns, methods):
        kwargs = method_params[method].copy()
        kwargs['labels'] = [index]
        kwargs['mode'] = mode
        if method == 'LIME':
            kwargs['_feature_names']=data[:1].columns.to_list()

        func = explain_tabular_dispatcher[method]
        

        with col:
            with st.spinner(f'Running {method}'):
                relevances = func(serialized_model, selected_data, training_data, **kwargs)
            fig, _ = plot_tabular(x=relevances, y=data[:1].columns, num_features=10, show_plot=False)
            st.pyplot(fig)

    # add some white space to separate rows
    st.markdown('')


st.stop()
