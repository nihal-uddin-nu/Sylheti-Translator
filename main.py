"""Ignore for now -- this is where the main logic for the web app will go. The web app will call the inference function from inference.py to get the translation result, which will then be displayed on the web page."""
import sys  # noqa: I001

import pandas as pd

import translator.detect_script as ds
import translator.transliterate_latin_sylheti as tls

df = pd.read_csv('data/Syl2Ban.csv')

input_text = input()

script = ds.detect_script(input_text)

match script:
    case "Latin":
        text_to_translate = tls.transliterate(input_text)

    case "Bangla":
        text_to_translate = input_text

    case "Mixed":
        print("Mixed script detected. Please provide input entirely in either Latin or Bengali script.")
        sys.exit()

    case "Unknown":
        print("Unable to detect script. Please provide input in either English or Bangla.")
        sys.exit()

# TODO: implement inference in web app here