"""
test_preprocess.py
Basic unit tests for the preprocessing module.
Run: python -m pytest tests/
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import clean_text


def test_lowercase():
    assert clean_text("Hello World") == clean_text("hello world")

def test_removes_urls():
    result = clean_text("Visit https://example.com for more info")
    assert 'http' not in result and 'example' not in result

def test_removes_html():
    result = clean_text("<p>Hello <b>World</b></p>")
    assert '<' not in result and '>' not in result

def test_removes_numbers():
    result = clean_text("There are 42 apples")
    assert '42' not in result

def test_non_empty_output():
    result = clean_text("The quick brown fox jumps over the lazy dog")
    assert len(result) > 0

if __name__ == "__main__":
    test_lowercase()
    test_removes_urls()
    test_removes_html()
    test_removes_numbers()
    test_non_empty_output()
    print("✅ All tests passed!")
