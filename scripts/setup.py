from scripts.build_datasets import build_datasets
from scripts.build_tries import build_word_tries, build_sentence_tries

def main() -> None:
    build_datasets()
    build_word_tries()
    build_sentence_tries()

if __name__ == "__main__":
    main()