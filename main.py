from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from preprocess import *
from Encorder import *
from Decorder import *
from tools import *
from Attention import *
from Train import *


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 75000, pairs, input_lang, output_lang, print_every=5000)

evaluateRandomly(encoder1, attn_decoder1, pairs, input_lang, output_lang)

output_words, attentions = evaluate(
    encoder1, attn_decoder1, "je suis trop froid .", input_lang, output_lang)
# plt.matshow(attentions.numpy())
plt.matshow(attentions)

evaluateAndShowAttention("elle a cinq ans de moins que moi .", encoder1, attn_decoder1, input_lang, output_lang)

evaluateAndShowAttention("elle est trop petit .", encoder1, attn_decoder1, input_lang, output_lang)

evaluateAndShowAttention("je ne crains pas de mourir .", encoder1, attn_decoder1, input_lang, output_lang)

evaluateAndShowAttention("c est un jeune directeur plein de talent .", encoder1, attn_decoder1, input_lang, output_lang)

