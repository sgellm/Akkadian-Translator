from transformers import PreTrainedTokenizerFast
import random
from torch.optim.lr_scheduler import StepLR
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from model import EncoderRNN, DecoderRNN
import pickle

wrapped_combo_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="combined_tokenizer.json",
    bos_token="<sos>",
    eos_token="<eos>",
    pad_token="<pad>",
    unk_token="<unk>",
    mask_token="<mask>"
)

SOS_token = wrapped_combo_tokenizer.bos_token_to_id
EOS_token = wrapped_combo_tokenizer.eos_token_to_id

def mask_inputs(ids, mask_token_id, mask_prob=.15):
    masked_ids = []
    for id in ids:
        if random.random() < mask_prob:
            masked_ids.append(mask_token_id)
        else:
            masked_ids.append(id)

    return masked_ids

def tensorFromSentence(tokenizer, sentence, input=False):
    # Ensure special tokens like EOS/SOS are added if needed
    ids = tokenizer.encode(sentence, add_special_tokens=False)
    ids = mask_inputs(ids, tokenizer.mask_token_id)
    ids = ids + [EOS_token]
    return torch.tensor(ids, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(seq):
    ak_sentence, en_sentence = seq.split('\t')
    input_tensor = tensorFromSentence(wrapped_combo_tokenizer, en_sentence)  # English sentence
    target_tensor = tensorFromSentence(wrapped_combo_tokenizer, ak_sentence)  # Akkadian sentence
    return (input_tensor, target_tensor)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden() # initalized the encoder

    # zeroes encoder/decoder gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # input tensors from tensorsFromPair
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    # encoder forward pass
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # add SOS token to beginning of decoder input
    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    # decoder forward pass
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    # backward pass for whole Seq2Seq network
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# functions for timing

def asMinutes(s):
    h = math.floor(s / 3600)
    s -= h * 3600
    m = math.floor(s / 60)
    s -= m * 60
    return '%dh %dm %ds' % (h, m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

# function for plotting
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01,
               start_iter=1, encoder_optimizer=None, decoder_optimizer=None, plot_losses=None):
    try:
        start = time.time()
        if plot_losses is None:
            plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        if encoder_optimizer is None:
            encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        if decoder_optimizer is None:
            decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

        scheduler_encoder = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=8000, gamma=0.97)
        scheduler_decoder = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=8000, gamma=0.97)

        training_pairs = [tensorsFromPair(pair) for pair in model_data]
        criterion = nn.NLLLoss()

        for iter in range(start_iter, n_iters + 1):
            teacher_forcing_ratio = max(0.9 * math.exp(-iter / 10000), 0.3)

            training_pair = training_pairs[iter - 1]

            input_tensor = training_pair[0] # english
            target_tensor = training_pair[1] # akkadian

            loss = train(input_tensor, target_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio)
            print_loss_total += loss
            plot_loss_total += loss

            scheduler_encoder.step()
            scheduler_decoder.step()

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f, lr=%.4f' % (timeSince(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg,
                                            encoder_optimizer.param_groups[0]['lr']))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
    except Exception as e:
        print(e)
    finally:
        showPlot(plot_losses)

hidden_size = 64
encoder1 = EncoderRNN(wrapped_combo_tokenizer.vocab_size, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, wrapped_combo_tokenizer.vocab_size, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 75000, print_every=1000)

pickle.dump((encoder1, attn_decoder1), open('encoder-decoder.pkl', 'wb'))