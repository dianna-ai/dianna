# DIANNA dashboard

The DIANNA dashboard can be used for explanation of the behaviour of several ONNX models trained for the tasks and datasets presented in the DIANNA [tutorials](../../tutorials). 
The dashboard shows the visual explanation of a models' decision on a selected data item by a selected XAI method (explainer). I
t allows you to compare the results of different explainers, as well as explanations of the top ranked predicted labels. The dashboard was created using streamlit.

To open the dashboard, you can install dianna via `pip install -e '.[dashboard]'` and run:

```console
dianna-dashboard
```

or, from this directory:

```console
streamlit run Home.py
```

Open the link on which the app is running. Note that you are running the dashboard *only locally*. The data you use in the dashboard is your local data, and it is *not* uploaded to any server.

## How to use the dashboard

This [video](https://youtu.be/9VM5acip2s8) shows you how to use the DIANNA dashboard for some of the DIANNA tutorials. Similarly, you can use it with your own data, models and lables.

--------------------------------------------------------------------------
![image](https://github.com/user-attachments/assets/1a98920e-f75e-468c-bf1f-f6e8bd2273ad)
