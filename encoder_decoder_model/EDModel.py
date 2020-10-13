import torch
import torch.nn as nn

import onmt
import onmt.inputters
import onmt.modules
import onmt.utils


from itertools import chain
import onmt.translate

vocab_fields = torch.load("./OpenNMT-py/data/data.vocab.pt")

src_text_field = vocab_fields["src"].base_field
src_vocab = src_text_field.vocab
src_padding = src_vocab.stoi[src_text_field.pad_token]

tgt_text_field = vocab_fields["tgt"].base_field
tgt_vocab = tgt_text_field.vocab
tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]

emb_size = 100
rnn_size = 500

encoder_embeddings = onmt.modules.Embeddings(emb_size, len(src_vocab), word_padding_idx = src_padding)
encoder_embeddings.load_pretrained_vectors("./OpenNMT-py/data/embeddings.enc.pt")
encoder = onmt.encoders.RNNEncoder(hidden_size = rnn_size, num_layers = 1, rnn_type = "LSTM", bidirectional = True, embeddings = encoder_embeddings)

decoder_embeddings = onmt.modules.Embeddings(emb_size, len(tgt_vocab), word_padding_idx = tgt_padding)
decoder_embeddings.load_pretrained_vectors("./OpenNMT-py/data/embeddings.dec.pt")
decoder = onmt.decoders.decoder.InputFeedRNNDecoder(hidden_size = rnn_size, num_layers = 1, bidirectional_encoder = True, rnn_type = "LSTM", embeddings = decoder_embeddings)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = onmt.models.model.NMTModel(encoder, decoder)
model.to(device)

model.generator = nn.Sequential(nn.Linear(rnn_size, len(tgt_vocab)), nn.LogSoftmax(dim = 1)).to(device)

loss = onmt.utils.loss.NMTLossCompute(criterion = nn.NLLLoss(ignore_index = tgt_padding, reduction = "sum"), generator = model.generator)

lr = 0.01
torch_optimizer = torch.optim.SGD(model.parameters(), lr = lr)
optim = onmt.utils.optimizers.Optimizer(torch_optimizer, learning_rate = lr, max_grad_norm = 2)

train_data_file = "OpenNMT-py/data/data.train.0.pt"
valid_data_file = "OpenNMT-py/data/data.valid.0.pt"
train_iter = onmt.inputters.inputter.DatasetLazyIter(dataset_paths = [train_data_file], fields = vocab_fields, batch_size = 50, batch_size_multiple = 1, batch_size_fn = None, device = device, is_train = True, pool_factor = 1, repeat = True)
valid_iter = onmt.inputters.inputter.DatasetLazyIter(dataset_paths = [valid_data_file], fields = vocab_fields, batch_size = 10, batch_size_multiple = 1, batch_size_fn = None, device = device, is_train = False, pool_factor = 1, repeat = False)

report_manager = onmt.utils.ReportMgr(report_every = 50, start_time = None, tensorboard_writer = None)
trainer = onmt.Trainer(model = model, train_loss = loss, valid_loss = loss, optim = optim, report_manager = report_manager)
trainer.train(train_iter = train_iter, train_steps = 400, valid_iter = valid_iter, valid_steps = 200)

src_reader = onmt.inputters.str2reader["text"]
tgt_reader = onmt.inputters.str2reader["text"]
scorer = onmt.translate.GNMTGlobalScorer(alpha = 0.7, beta = 0., length_penalty = "avg", coverage_penalty = "none")
gpu = 0 if torch.cuda.is_available() else -1
translator = onmt.translate.Translator(model = model, fields = vocab_fields, src_reader = src_reader, tgt_reader = tgt_reader, global_scorer = scorer, gpu = gpu)
builder = onmt.translate.TranslationBuilder(data = torch.load(valid_data_file), fields = vocab_fields)

for batch in valid_iter:
    trans_batch = translator.translate_batch(batch = batch, src_vocabs = [src_vocab], attn_debug = False)
    translations = builder.from_batch(trans_batch)
    for trans in translations:
        print(trans.log(0))