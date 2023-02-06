import argparse
import time
import numpy as np
import math
import torch
import torch.nn as nn
import torch.onnx

import data
from utils import get_device, repackage_hidden, make_reproducible
from rnnlm import RNNModel
import gensim


def init_glove_embeddings(model: RNNModel, gensim_glove_weights, vocab, freeze_glove):
    # TODO: implement this function
    eos_token_embedding = torch.tensor([np.random.uniform(low=-1, high=1.0, size=(50,))])
    unk_token_embedding = torch.tensor([np.random.uniform(low=-1, high=1.0, size=(50,))])
    pre_trained_embeddings = torch.zeros([len(vocab.type2index.keys()), 50])
    for word, idx in vocab.type2index.items():
        if word == "<eos>":
            pre_trained_embeddings[idx] = eos_token_embedding
        elif word == "<unk>":
            pre_trained_embeddings[idx] = unk_token_embedding
        elif gensim_glove_weights.has_index_for(word):
            # if the word is present in pre-trained embedding include with its pre-trained embedding
            pre_trained_embeddings[idx] = torch.tensor(gensim_glove_weights.get_vector(gensim_glove_weights.key_to_index.get(word), norm=True))
        else:
            # if the word is not present in pre-trained embedding include with its unk token embeddings
            pre_trained_embeddings[idx] = unk_token_embedding

    with torch.no_grad():
        model.in_embedder.weight.data.copy_(torch.FloatTensor(pre_trained_embeddings))
        model.in_embedder.weight.requiresGrad = not freeze_glove


def compute_perplexity(loss: float):
    # # TODO: implement this function
    # raise NotImplementedError

    return np.exp(loss)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="./data/wikitext-2-mini",
        help="location of the data corpus",
    )

    parser.add_argument(
        "--emsize", type=int, default=50, help="size of word embeddings"
    )
    parser.add_argument(
        "--nhid", type=int, default=200, help="number of hidden units per layer"
    )
    parser.add_argument("--nlayers", type=int, default=2, help="number of layers")
    
    parser.add_argument("--rnn_type", type=str, default="elman", help="type of rnn")
    
    parser.add_argument("--use_glove", type=int, default=0, help="use glove 50 embeddings")
    parser.add_argument("--freeze_glove", type=int, default=1, help="freeze glove 50 embeddings")
    

    parser.add_argument("--bidirectional", type=int, default=0, help="directionality of rnn 0/1")
    
    parser.add_argument("--lr", type=float, default=20, help="initial learning rate")
    parser.add_argument("--clip", type=float, default=0.25, help="gradient clipping")
    parser.add_argument("--epochs", type=int, default=40, help="upper epoch limit")
    parser.add_argument(
        "--batch_size", type=int, default=20, metavar="N", help="batch size"
    )
    parser.add_argument("--seq-len", type=int, default=35, help="sequence length")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="dropout applied to layers (0 = no dropout)",
    )
    parser.add_argument("--adam", type=int, default=0, help="use adam 0/1")
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument(
        "--log-interval", type=int, default=200, metavar="N", help="report interval"
    )
    parser.add_argument(
        "--save", type=str, default="model.pt", help="path to save the final model"
    )
    return parser.parse_args()


# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.


def batchify(data, batch_size):
    # Work out how cleanly we can divide the dataset into batch_size parts.
    n_batches = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, n_batches * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.


def get_batch(source, i):
    seq_len = min(args.seq_len, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    return data, target


def evaluate(model, data_source, criterion):
    model.eval()
    total_loss = 0.0
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.seq_len):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train_model_step(corpus, args, model, criterion, epoch, lr):
    # Turn on training mode which enables dropout.
    train_data = batchify(corpus.train, args.batch_size)
    model.train()
    total_loss = 0.0
    start_time = time.time()
    ntokens = len(corpus.vocab)
    hidden = model.init_hidden(args.batch_size)
    # added adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) if args.adam else None
    
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.seq_len)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if args.adam:
            optimizer.zero_grad()
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # params to update
        params_to_update = [p for p in model.parameters() if p.requires_grad == True]

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(params_to_update, args.clip)
        
        # take a step in the adam optimizer
        if args.adam:
            optimizer.step()
        else:
            for p in params_to_update:
                p.data.add_(p.grad, alpha=-lr)
        
        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | "
                "loss {:5.2f} | ppl {:8.2f}".format(
                    epoch,
                    batch,
                    len(train_data) // args.seq_len,
                    lr,
                    elapsed * 1000 / args.log_interval,
                    cur_loss,
                    compute_perplexity(cur_loss),
                )
            )
            total_loss = 0
            start_time = time.time()
    # Loop over epochs.


def train_model(corpus, args, model, criterion):
    best_val_loss = None
    eval_batch_size = 10
    val_data = batchify(corpus.valid, eval_batch_size)
    lr = args.lr
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_model_step(corpus, args, model, criterion, epoch=epoch, lr=lr)
            val_loss = evaluate(model, val_data, criterion)
            print("-" * 89)
            print(
                "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
                "valid ppl {:8.2f}".format(
                    epoch,
                    (time.time() - epoch_start_time),
                    val_loss,
                    compute_perplexity(val_loss),
                )
            )
            print("-" * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, "wb") as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")


def test_model(corpus, args, model, criterion):  # Load the best saved model.
    model = RNNModel.load_model(args.save)
    eval_batch_size = 10
    test_data = batchify(corpus.test, eval_batch_size)
    test_loss = evaluate(model, test_data, criterion=criterion)
    print("=" * 89)
    print(
        "| End of training | test loss {:5.2f} | test ppl {:8.2f}".format(
            test_loss, math.exp(test_loss)
        )
    )
    print("=" * 89)


if __name__ == "__main__":
    args = parse_args()
    make_reproducible(args.seed)
    device = get_device()
    print(device)
    corpus = data.Corpus(args.data)
    print(corpus.train)
    count_of_all_tokens = sum([_count for _word, _count in corpus.train_word_counts.items()])
    count_of_unk_tokens = corpus.train_word_counts["<unk>"]
    print("Count of all tokens: {}".format(count_of_all_tokens))
    print("Count of <unk> tokens: {}".format(count_of_unk_tokens))
    print("Percentage of <unk> tokens: {}".format(float(count_of_unk_tokens * 100)/count_of_all_tokens))
    
    eval_batch_size = 10

    test_data = batchify(corpus.test, eval_batch_size)

    ntokens = len(corpus.vocab)
    
    path_to_glove_embeddings_in_gensim_word2vec_format = "/soe/vigneshs/projects/nlp_244/NLP244-quest1/glove/glove-word2vec.6B.50d.txt"
    
    use_glove = True if args.use_glove else False
        
    model = RNNModel(ntokens, args.emsize, args.nhid, args.nlayers, args.bidirectional, args.dropout, args.rnn_type, use_glove).to(device)

    if use_glove:
        gensim_glove_weights = gensim.models.KeyedVectors.load_word2vec_format(path_to_glove_embeddings_in_gensim_word2vec_format)
        freeze_glove = True if args.freeze_glove else False
        init_glove_embeddings(model=model, gensim_glove_weights=gensim_glove_weights, vocab=corpus.vocab, freeze_glove=freeze_glove)


    criterion = nn.NLLLoss()
    train_model(corpus, args, model, criterion)
    test_model(corpus, args, model, criterion)
