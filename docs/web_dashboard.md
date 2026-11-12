# Dashboard in your browser

The [DIANNA dashboard](https://github.com/dianna-ai/dianna/tree/main/dianna/dashboard) also runs entirely in your
browser, without installing anything:

<p><a href="dashboard/"><b>Open the dashboard</b></a></p>

It is the same dashboard as `dianna-dashboard`, with Python, DIANNA and the models running in your browser
([stlite](https://github.com/whitphx/stlite), [Pyodide](https://pyodide.org) and
[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/)). The data and models you use are not uploaded anywhere.

Good to know:

- Loading Python and the packages takes about half a minute.
- It needs a browser with JavaScript Promise Integration (JSPI), e.g. a recent Chrome or Edge.
- The explainers run slower than in the installed dashboard.
- The *Nature of EU laws* text example is not available: its model needs PyTorch and XGBoost, which do not run in the
  browser.
