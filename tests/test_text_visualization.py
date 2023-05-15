import os
import re
import shutil
import unittest
from pathlib import Path
from dianna.visualization.text import highlight_text

tokenizer = re.compile(r'(\w+|\S)')


class TextExample:
    """Text and explanation for running visualizing tests."""
    original_text = 'Doloremque aliquam totam ut. Aspernatur repellendus autem quia deleniti. Natus accusamus ' \
                    'doloribus et in quam officiis veniam et. '

    tokens = tokenizer.findall(original_text)

    explanation = [('aliquam', 1, 0.6450221), ('et', 19, 0.63976675),
                   ('in', 15, 0.6397511), ('Aspernatur', 5, 0.6329795),
                   ('totam', 2, 0.6242337), ('quia', 8, 0.61881626),
                   ('quam', 16, 0.6173148), ('Natus', 11, 0.60527676),
                   ('ut', 3, 0.60203224)]


class TextExampleWithExpectedHtml:
    """Short text and explanation and its expected html output after visualizing."""
    expected_html = (
        '<mark style="background-color: hsl(0, 100%, 63%, 0.8); line-height:1.75">Such</mark> '
        '<mark style="background-color: hsl(0, 100%, 62%, 0.8); line-height:1.75">a</mark> '
        '<mark style="background-color: hsl(0, 100%, 50%, 0.8); line-height:1.75">bad</mark> '
        '<mark style="background-color: hsl(0, 100%, 61%, 0.8); line-height:1.75">movie</mark> '
        '<mark style="background-color: hsl(0, 0%, 75%, 0.8); line-height:1.75">.</mark>\n'
    )

    original_text = 'Such a bad movie.'

    tokens = tokenizer.findall(original_text)

    explanation = [('bad', 2, 0.9959058), ('movie', 3, 0.78263557),
                   ('a', 1, 0.7753202), ('Such', 0, 0.73788315)]


class TextVisualizationTestCase(unittest.TestCase):
    """Suite of tests for visualizing text given text and explanation data."""
    temp_folder = 'temp_text_visualization_test'
    html_file_path = str(Path(temp_folder) / 'output.html')

    def test_text_visualization_html_output_exists(self):
        """Test if any output is generated at all."""
        highlight_text(TextExample.explanation,
                       TextExample.tokens,
                       output_html_filename=self.html_file_path)

        assert Path(self.html_file_path).exists()

    def test_text_visualization_html_output_contains_text(self):
        """Test if all words in the input are present in the output html."""
        highlight_text(TextExample.explanation,
                       TextExample.tokens,
                       output_html_filename=self.html_file_path)

        assert Path(self.html_file_path).exists()
        with open(self.html_file_path, encoding='utf-8') as result_file:
            result = result_file.read()
        for word in TextExample.tokens:
            assert word in result

    def test_text_visualization_html_output_is_correct(self):
        """Test if exact html output of visualization is correct."""
        highlight_text(TextExampleWithExpectedHtml.explanation,
                       TextExampleWithExpectedHtml.tokens,
                       output_html_filename=self.html_file_path)

        assert Path(self.html_file_path).exists()

        with open(self.html_file_path, encoding='utf-8') as result_file:
            result = result_file.read()

        assert result == TextExampleWithExpectedHtml.expected_html

    def test_text_visualization_show_plot(self):
        """Test if it runs while showing the plot."""
        highlight_text(TextExample.explanation,
                       TextExample.tokens,
                       show_plot=True)

    def setUp(self) -> None:
        """Setup."""
        os.mkdir(self.temp_folder)

    def tearDown(self) -> None:
        """Tear down."""
        shutil.rmtree(self.temp_folder, ignore_errors=True)
