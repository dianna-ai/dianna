# DIANNA dashboard

The DIANNA dashboard (powered by [streamlit](https://streamlit.io/) <img width="25" alt="Streamlit Logo" src="https://github.com/user-attachments/assets/2cac7d5d-c11a-48fe-b58e-71b15baaa163">) can be used for explanation of ONNX models trained for the tasks and datasets presented in several (marked with <img width="25" alt="Streamlit Logo" src="https://github.com/user-attachments/assets/2cac7d5d-c11a-48fe-b58e-71b15baaa163">) of the DIANNA [tutorials](../../tutorials/README.md). 
The dashboard shows the visual explanation of a models' decision on a selected data item by a selected XAI method (explainer). Results of different explainers can be compared, as well as explanations of the top predicted labels. 

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

This [video](https://youtu.be/9VM5acip2s8) shows you how to use the DIANNA dashboard for some of the <img width="25" alt="Streamlit Logo" src="https://github.com/user-attachments/assets/2cac7d5d-c11a-48fe-b58e-71b15baaa163"> [tutorials](../../tutorials/README.md#summary-of-all-tutorials) to interactively explore the use cases from some of DIANNA's . 
Similarly, you can use it with your own data, models and lables.

--------------------------------------------------------------------------
![image](https://github.com/user-attachments/assets/1a98920e-f75e-468c-bf1f-f6e8bd2273ad)
