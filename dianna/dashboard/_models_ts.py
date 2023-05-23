import numpy as np
import onnxruntime as ort
import dianna


def predict(*, model, ts_data):
    # model must receive data in the order of [batch, timeseries, channels]
    # data = data.transpose([0,2,1])
    # get ONNX predictions
    sess = ort.InferenceSession(model)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    onnx_input = {input_name: ts_data.astype(np.float32)}
    pred_onnx = sess.run([output_name], onnx_input)[0]

    return pred_onnx


def _run_rise_timeseries(_model, ts_data, **kwargs):

    def run_model(ts_data):
        return predict(model=_model, ts_data=ts_data)

    explanation = dianna.explain_timeseries(
        run_model,
        timeseries_data=ts_data[0],
        method='RISE',
        **kwargs,
    )

    return explanation


def _run_lime_timeseries(_model, ts_data, **kwargs):

    def run_model(ts_data):
        return predict(model=_model, ts_data=ts_data)

    explanation = dianna.explain_timeseries(
        run_model,
        ts_data[0],
        method='LIME',
        num_features=len(ts_data[0]),
        num_slices=len(ts_data[0]),
        num_samples=100,
        distance_method='dtw',
        **kwargs,
    )

    return explanation


explain_ts_dispatcher = {
    'RISE': _run_rise_timeseries,
    'LIME': _run_lime_timeseries,
}
