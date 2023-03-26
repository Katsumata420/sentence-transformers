"""
This file loads sentences from a provided text file. It is expected, that the there is one sentence per line in that text file.
SimCSE will be training using these sentences. Checkpoints are stored every 500 steps to the output folder.
Usage:
python train_simcse_from_file.py path/to/sentences.txt
"""
import argparse
import json
import logging
import math
from typing import List

from torch.utils.data import DataLoader
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, InputExample


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-batch-size", default=128, type=int)
    parser.add_argument("--max-seq-length", default=32, type=int)
    parser.add_argument("--n-epoch", default=1, type=int)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def load_train_data(file_path: str) -> List[InputExample]:
    train_samples = []
    with open(file_path) as i_f:
        for line in i_f:
            line = line.strip()
            data = json.loads(line)
            train_samples.append(InputExample(texts=[data["phrase"], data["phrase"]]))
            # train_samples.append(InputExample(texts=[line, line]))
    return train_samples


def main():
    args = get_args()

    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(args.model_name_or_path, max_seq_length=args.max_seq_length)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    ################# Read the train corpus  #################
    train_samples = load_train_data(args.input_file)
    logging.info("Train sentences: {}".format(len(train_samples)))

    # We train our model using the MultipleNegativesRankingLoss
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size, drop_last=True)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = math.ceil(len(train_dataloader) * args.n_epoch * 0.1)  # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=args.n_epoch,
              warmup_steps=warmup_steps,
              optimizer_params={'lr': 5e-5},
              checkpoint_path=args.output_path,
              show_progress_bar=True,
              use_amp=False  # Set to True, if your GPU supports FP16 cores
              )


if __name__ == "__main__":
    main()
