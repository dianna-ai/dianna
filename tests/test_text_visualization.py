import os
import shutil
import unittest
from pathlib import Path
import spacy
from dianna.visualization.text import highlight_text
from dianna.utils.tokenizers import SpacyTokenizer


spacy.cli.download('en_core_web_sm')


class TextExample:
    """Text and explanation for running visualizing tests."""
    original_text = 'Doloremque aliquam totam ut. Aspernatur repellendus autem quia deleniti. Natus accusamus ' \
                    'doloribus et in quam officiis veniam et. '
    tokenizer = SpacyTokenizer()
    tokens = tokenizer.tokenize(original_text)
    explanation = [
                    ('aliquam', 1, 0.6450221),
                    ('et', 19, 0.63976675),
                    ('in', 15, 0.6397511),
                    ('Aspernatur', 5, 0.6329795),
                    ('totam', 2, 0.6242337),
                    ('quia', 8, 0.61881626),
                    ('quam', 16, 0.6173148),
                    ('Natus', 11, 0.60527676),
                    ('ut', 3, 0.60203224)
                    ]


class TextExampleWithExpectedHtml:
    """Short text and explanation and its expected html output after visualizing."""
    expected_html = '<html><body><span style="background:rgba(255, 0, 0, 0.59)">Such</span> ' \
                    '<span style="background:rgba(255, 0, 0, 0.62)">a</span> ' \
                    '<span style="background:rgba(255, 0, 0, 0.80)">bad</span> ' \
                    '<span style="background:rgba(255, 0, 0, 0.63)">movie</span> ' \
                    '<span style="background:rgba(128, 128, 128, 0.3)">.</span></body></html>\n'

    original_text = 'Such a bad movie.'
    tokenizer = SpacyTokenizer()
    tokens = tokenizer.tokenize(original_text)
    explanation =[
                    ('bad', 2, 0.9959058),
                    ('movie', 3, 0.78263557),
                    ('a', 1, 0.7753202),
                    ('Such', 0, 0.73788315)
                    ]


class TextVisualizationTestCase(unittest.TestCase):
    """Suite of tests for visualizing text given text and explanation data."""
    temp_folder = 'temp_text_visualization_test'
    html_file_path = str(Path(temp_folder) / 'output.html')

    def test_text_visualization_html_output_exists(self):
        """Test if any output is generated at all."""
        highlight_text(TextExample.explanation, TextExample.tokens,
                       output_html_filename=self.html_file_path)

        assert Path(self.html_file_path).exists()

    def test_text_visualization_html_output_contains_text(self):
        """Test if all words in the input are present in the output html."""
        highlight_text(TextExample.explanation, TextExample.tokens,
                       output_html_filename=self.html_file_path)

        assert Path(self.html_file_path).exists()
        with open(self.html_file_path, encoding='utf-8') as result_file:
            result = result_file.read()
        for word in TextExample.tokens:
            assert word in result

    def test_text_visualization_html_output_is_correct(self):
        """Test if exact html output of visualization is correct."""
        highlight_text(TextExampleWithExpectedHtml.explanation, TextExampleWithExpectedHtml.tokens,
                       output_html_filename=self.html_file_path)

        assert Path(self.html_file_path).exists()

        with open(self.html_file_path, encoding='utf-8') as result_file:
            result = result_file.read()
        assert result == TextExampleWithExpectedHtml.expected_html

    def test_text_visualization_show_plot(self):
        """Test if it runs while showing the plot."""
        highlight_text(TextExample.explanation, TextExample.tokens,
                       show_plot=True)

    def setUp(self) -> None:
        os.mkdir(self.temp_folder)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_folder, ignore_errors=True)
