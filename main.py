import pandas as pd
import sentencepiece as spm
import detect_script as ds
import transliterate_latin_sylheti as tls

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

# implement sylheti to bangla translsation here (train model using 5k pair dataset?)

# implement argos-translate bn to en here