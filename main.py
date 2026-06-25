"""Ignore for now -- this is where the main logic for the web app will go. The web app will call the inference function from inference.py to get the translation result, which will then be displayed on the web page."""
import pandas as pd
import sentencepiece as spm
import sylheti_bangla_model.detect_script as ds
import sylheti_bangla_model.transliterate_latin_sylheti as tls

df = pd.read_csv('data/Syl2Ban.csv')

input_text = input()

script = ds.detect_script(input_text)

match script:
    case "English":
        text_to_translate = tls.transliterate(input_text)

    case "Bangla":
        text_to_translate = input_text

    case "Mixed":
        print("Mixed script detected. Please provide input in either English or Bangla.")
        exit()

    case "Unknown":
        print("Unable to detect script. Please provide input in either English or Bangla.")
        exit()

# TODO: implement inference in web app here