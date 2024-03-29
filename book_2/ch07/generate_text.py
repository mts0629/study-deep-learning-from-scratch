import sys
sys.path.append("..")

from rnnlm_gen import RnnlmGen
from dataset import ptb


def main():
    corpus, word_to_id, id_to_word = ptb.load_data("train")
    vocab_size = len(word_to_id)
    corpus_size = len(corpus)

    model = RnnlmGen()
    model.load_params("../ch06/Rnnlm.pkl")  # Load pre-trained parameters

    # Set start/skip words
    start_word = "you"
    start_id = word_to_id[start_word]
    skip_words = ["N", "<unk>", "$"]
    skip_ids = [word_to_id[w] for w in skip_words]

    # Generate a sentence
    word_ids = model.generate(start_id, skip_ids)
    txt = " ".join([id_to_word[i] for i in word_ids])
    txt = txt.replace(" <eos>", ".\n")
    print(txt)


if __name__ == "__main__":
    main()
