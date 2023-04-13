# DIANNA dashboard


A dashboard was created for DIANNA using [streamlit](https://streamlit.io/) that can be used for simple exploration of your trained model explained by DIANNA. The dashboard produces the visual explanation of your selected XAI method. Additionally it allows you to compare the results of different XAI methods, as well as explanations of the top ranked predicted labels.

To open the dashboard via the cli anywhere on your system, run

```console
dianna-dashboard
```

Or from the dashboard directory via streamlit directly:

```console
streamlit run Home.py
```

Open the link on which the app is running. Note that you are running the dashboard *only locally*. The data you use in the dashboard is your local data, and it is *not* uploaded to any server.


## How to use the dashboard

The dashboard will automatically open in the welcome page tab. In the sidebar you can open the image or text pages.

### Images

- Click on select image and select an image saved on your computer for which you want the DIANNA explanation.
- Click on select model and select your trained model used for the prediction and explanation.
- Click on select labels and select a text file containing the labels of your trained model. Labels should be separated by line breaks in the file, no other separators should be used. Make sure that both the <ins>labels</ins> and <ins>ordering of the labels</ins> is correct in correspondence to your trained model.
- Check the XAI methods you want to use, multiple methods can be used at the same time for comparison.
- Your image explanation will start loading.

Additionally, you can:
- Select the number of top results you which to show the explanation for, i.e. if you use 2, the dashboard will show not only the explanation for the main prediction for the selected image, but also for the second most likely prediction.
- Set the method specific settings to the selected XAI methods. If you don't change these, the default values are used.

### Text

- Input the text you want to use in the "Input string" box. Press enter to update the explanation.
- Click on select model and select your trained model used for the prediction and explanation.
- Click on select labels and select a text file containing the labels of your trained model. Labels should be separated by line breaks in the file, no other separators should be used. Make sure that both the <ins>labels</ins> and <ins>ordering of the labels</ins> is correct in correspondence to your trained model.
- Check the XAI methods you want to use, multiple methods can be used at the same time for comparison.
- Your image explanation will start loading.

Additionally, you can set the method specific settings to the selected XAI methods. If you don't change these, the default values are used.
