# DIANNA dashboard

A dashboard was created for DIANNA using [plotly dash](https://plotly.com/dash/) that can be used for simple exploration of your trained model explained by DIANNA. The dashboard produces the visual explanation of your selected XAI method. Additionally it allows you to compare the results of different XAI methods, as well as explanations of the top ranked predicted labels.

To use the dashboard, run 
```console
python3 ./dashboard/dashboard.py
```
and open the link on which Dash is running. Note that you are running the dashboard only locally. The data you use in the dashboard is your local data, and it is *not* uploaded to any server.


The dashboard will automatically open in the Image tab, where image explanations can be performed. To use the text explainer, switch to the text tab.

## Images
- Click on select image and select an image saved onb your computer.


## Text