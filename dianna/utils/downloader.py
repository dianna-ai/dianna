"""DIANNA file downloader.

Instructions for adding a new file:

There are three dictionaries defining the available files:
one for models, one for labels, one for data.
Each entry has the (output) filename as key, and a list with the URL and sha256 hash as value.
The sha256 hash can be obtained locally with `sha256sum <filename>`. Alternatively,
set the hash to None and download the file with the `download` function.
the sha256 sum will be printed to the command line and can be inserted into this file.
Do not forget to prefix it with `sha256:`
"""
import pooch

__all__ = ["list_available_files", "download"]

models = {
    "mnist_model.onnx": [
        "doi:10.5281/zenodo.5907176/mnist_model.onnx",
        "sha256:8cc46d73a607f8cab0ba5cae30708f496c8316571bdb5dac93ac5bff1c452aff"
    ],
    "coffee.onnx": [
        "doi:10.5281/zenodo.10579457/coffee.onnx",
        "sha256:3734b600a88c82b33226675b38cb56e4d2fcd003c296414c7def8c079f795dba"
    ],
    "penguin_model.onnx": [
        "doi:10.5281/zenodo.10580742/penguin_model.onnx",
        "sha256:907637d5769c221878baf6bb5c3bf7c716880d53f544e6ce197368d8dfba78c6"
    ],
    "geometric_shapes_model.onnx": [
        "doi:10.5281/zenodo.5907058/geometric_shapes_model.onnx",
        "sha256:919b878a759151111fb273cebc4e32de2e133b2993e7f1402645db4e13e9bfc9"
    ],
    "mnist_model_tf.onnx": [
        "https://github.com/dianna-ai/dianna/raw/main/dianna/models/mnist_model_tf.onnx",
        "sha256:ec944fa3ea327a9232451d7bd368e4fe3dea55aacae3c8dbaab1ee8b38873482"
    ],
    "season_prediction_model_temp_max_binary.onnx": [
        "https://github.com/dianna-ai/dianna/raw/main/dianna/models/season_prediction_model_temp_max_binary.onnx",
        "sha256:68335f069b673b90fac3b879d38b1471e05c1c1ca64b9045eb0d383534c2986a"
    ],
    "season_prediction_model.onnx": [
        "doi:10.5281/zenodo.7543882/season_prediction_model.onnx",
        "sha256:38395a100f0379d11e7249c1491e8e4735e2704ed07b747d71431b6e572a732a"
    ],
    "leafsnap_model.onnx": [
        "doi:10.5281/zenodo.5907195/leafsnap_model.onnx",
        "sha256:8cefd92fee4b5e7f3bb94843c8504bb83a84bed38a28e808fe79028c8078c156"
    ],
    "movie_review_model.onnx": [
        "doi:10.5281/zenodo.5910597/movie_review_model.onnx",
        "sha256:f38fcd83a02f08fc3fc94b6dbcbfdec22281a7fad4c6d22112556b9c564ca6d2"
    ],
    "sunshine_hours_regression_model.onnx": [
        "doi:10.5281/zenodo.10580832/sunshine_hours_regression_model.onnx",
        "sha256:65904d98fc281c3d2604b646d2c85eb6ebe710340fb01466fb3d571a51810c7e"
    ],
    "apertif_frb_dynamic_spectrum_model.onnx": [
        "doi:10.5281/zenodo.10656613/apertif_frb_dynamic_spectrum_model.onnx",
        "sha256:3c87db3c6257d7f251a7bdceb3197d5bb482b8edc19870219fb7ca7c204dd257"
    ],
}

labels = {
    "apertif_frb_classes.txt": [
        "https://github.com/dianna-ai/dianna/raw/main/dianna/labels/apertif_frb_classes.txt",
        "sha256:7df809e9f028e59021c819408c2e3d06c7c1903b1d45c05847b0d22a6d8d43e2"
    ],
    "labels_mnist.txt": [
        "https://github.com/dianna-ai/dianna/raw/main/dianna/labels/labels_mnist.txt",
        "sha256:9b38d284bae4dc26e593ab0d5cb50d846c3328a8c0467291a8a473cf15c1615b"
    ],
    "labels_resnet50.txt": [
        "https://github.com/dianna-ai/dianna/raw/main/dianna/labels/labels_resnet50.txt",
        "sha256:734ac9d9fdc5b3594443cca61021e5b9eb96e0473c607ed03d8810c63fe48291"
    ],
    "labels_text.txt": [
        "https://github.com/dianna-ai/dianna/raw/main/dianna/labels/labels_text.txt",
        "sha256:52daeeabd704041e73d80db62e3812265dbef4add66ad8b8cedb03099439f7b9"
    ],
    "leafsnap_classes.csv": [
        "https://github.com/dianna-ai/dianna/raw/main/dianna/labels/leafsnap_classes.csv",
        "sha256:f115a084e8d88d490a1d216d27f2b57d46d348513aa41f51303143f0a2d101a7"
    ],
    "weather_data_labels.txt": [
        "https://github.com/dianna-ai/dianna/raw/main/dianna/labels/weather_data_labels.txt",
        "sha256:08926fa4a754a6536d46ccb865bb07a4910b4cf71911434cc2ff8a4c7466ce34"
    ],
}

data = {
    "FRB211024.npy": [
        "https://github.com/dianna-ai/dianna/raw/main/dianna/data/FRB211024.npy",
        "sha256:9bb567d2e2b9b1960f5f83c5f5c2539b38889de19784650c70e53ac3753f5153"
    ],
    "bee.jpg": [
        "https://github.com/dianna-ai/dianna/raw/main/dianna/data/bee.jpg",
        "sha256:3d829b4f5173363ec960c0b4de5130c96e4ff9c89f7cd5a784947150ffc119e6"
    ],
    "binary-mnist.npz": [
        "https://github.com/dianna-ai/dianna/raw/main/dianna/data/binary-mnist.npz",
        "sha256:922f97603522504808deaaa144af7594454eb3cf048917fc1f88de0cd0012add"
    ],
    "coffee_test.csv": [
        "https://github.com/dianna-ai/dianna/raw/main/dianna/data/coffee_test.csv",
        "sha256:251c56a8d24abe1416f545907fb37cbbe0a03e92d6ad8e4b89641980289947de"
    ],
    "coffee_train.csv": [
        "https://github.com/dianna-ai/dianna/raw/main/dianna/data/coffee_train.csv",
        "sha256:34cc37eff7310b33a7886a3d4bee42074126cb72287b39c58b369836bf56281e"
    ],
    "digit0.jpg": [
        "https://github.com/dianna-ai/dianna/raw/main/dianna/data/digit0.jpg",
        "sha256:2193cbb0ec58dbe4574b53476ed8ea15fdae09810a4fad7754e39954f252572a"
    ],
    "digit1.png": [
        "https://github.com/dianna-ai/dianna/raw/main/dianna/data/digit1.png",
        "sha256:3a9b5fa125dc51e3f9afef3482b6dd540e3851032a172b50557d5aaacc510f84"
    ],
    "leafsnap_example_acer_rubrum.jpg": [
        "https://github.com/dianna-ai/dianna/raw/main/dianna/data/leafsnap_example_acer_rubrum.jpg",
        "sha256:2b34227bbc4b7826e1e7743e2c376edbd72ee8bc5b3894711c0a81e2acd2e412"
    ],
    "movie_reviews_word_vectors.txt": [
        "https://github.com/dianna-ai/dianna/raw/main/dianna/data/movie_reviews_word_vectors.txt",
        "sha256:52ac88e925df1d06402b6b57c9cb48864343f0d942e8544267d2acc89c8b8d50"
    ],
    "shapes.npz": [
        "doi:10.5281/zenodo.5012824/shapes.npz",
        "sha256:58a644566482f5780b0e7132b3bcecfaf549ebef615f10912ba746a91ef588e1"
    ],
    "weather_data.npy": [
        "https://github.com/dianna-ai/dianna/raw/main/dianna/data/weather_data.npy",
        "sha256:d848dfc1effc958ecf73a32134c582944f32f987c2924eebbd47a5020417a303"
    ],
    "weather_prediction_dataset.csv": [
        "doi:10.5281/zenodo.4770936/weather_prediction_dataset.csv",
        "sha256:296220187f8eb8f1b3d0138dfd121683b1daa36983fed5e547f59e0d93c3d004"
    ],
    "weather_prediction_dataset_light.csv": [
        "doi:10.5281/zenodo.5071376/weather_prediction_dataset_light.csv",
        "sha256:945883204cae14251aec44e78845eb880ed3c04ac56a26fb5283c20f149be960"
    ],
}


def list_available_files():
    """Lists DIANNA-related model, data, and label files available for download."""
    print("Available model files:")
    for item in models.keys():
        print(item)
    print()

    print("Available label files:")
    for item in labels.keys():
        print(item)
    print()

    print("Available data files:")
    for item in data.keys():
        print(item)
    print()


def download(file, file_type, path=pooch.os_cache("dianna")):
    """Download a file.

    Args:
        file (str): name of the file. Use list_available_files() to get an overview
        file_type (str): model, data, or label
        path (str): download directory (default: OS cache directory)

    Returns:
        Full path to downloaded file (str)
    """
    if file_type == "model":
        files = models
    elif file_type == "label":
        files = labels
    elif file_type == "data":
        files = data
    else:
        raise ValueError(
            f"file_type must be model, label, or data. Got {file_type}")

    try:
        url, known_hash = files[file]
    except KeyError:
        raise KeyError(f"{file} is not a known file. "
                       "Run list_available_files() "
                       "to get a list of available files")
    local_file = pooch.retrieve(url,
                                fname=file,
                                known_hash=known_hash,
                                path=path)
    return local_file
