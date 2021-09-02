import numpy as np
import onnxruntime as ort
from scipy.special import expit
from torchtext.data import get_tokenizer
from torchtext.vocab import Vectors
import dianna
import dianna.visualization
import spacy


class ModelRunner():
    def __init__(self, model_path, word_vector_file, max_filter_size):
        self.filename = model_path
        # ensure the spacy english is downloaded
        spacy.cli.download('en_core_web_sm')
        self.tokenizer = get_tokenizer('spacy', 'en_core_web_sm')
        self.vocab = Vectors(word_vector_file, cache='.')

        self.max_filter_size = max_filter_size

    def __call__(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]

        output = []

        sess = ort.InferenceSession(self.filename)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        for sentence in sentences:
            # get tokens
            tokens = self.tokenizer(sentence)
            if len(tokens) < self.max_filter_size:
                tokens += ['<pad>'] * (self.max_filter_size - len(tokens))

            # numericalize
            tokens = [self.vocab.stoi[token] if token in self.vocab.stoi else self.vocab.stoi['<unk>'] for token in tokens]
            # feed to model
            onnx_input = {input_name: [tokens]}
            pred = expit(sess.run([output_name], onnx_input)[0])

            # output 2 classes
            positivity = pred[:, 0]
            negativity = 1 - positivity

            output.append(np.transpose([negativity, positivity])[0])

        return np.array(output)


def test_lime_text():
    model_path = 'tests/test_data/movie_review_model.onnx'
    word_vector_file = 'tests/test_data/word_vectors.txt'
    runner = ModelRunner(model_path, word_vector_file, max_filter_size=5)

    review = 'this was a bad movie'

    explanation = dianna.explain(runner, review, method='LIME', random_state=42)
    words = [element[0] for element in explanation]
    scores = [element[1] for element in explanation]

    expected_words = ['bad', 'movie', 'was', 'a', 'this']
    expected_scores = [-.304, -.039, -.039, -.035, .003]
    assert words == expected_words
    assert np.allclose(scores, expected_scores, atol=.01)
