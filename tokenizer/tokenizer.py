import sentencepiece as spm
import os
import sys

def main():

    # Get the parent directory
    parent_dir = os.path.dirname(os.path.realpath(__file__))

    # Add the parent directory to sys.path
    sys.path.append(parent_dir)

    spm.SentencePieceTrainer.Train(input="data/unsupervised-pretraining/training.txt",
                                   model_prefix="tweets",
                                   vocab_size=16000,
                                   character_coverage=1.0)
    
if __name__ == "__main__":
    main()