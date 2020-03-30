# Model is straight from
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
import torch

import model.encoder
import model.decoder


def load(directory_name):
    """
    Loads model to directory directory_name
    Be sure to move it to the right device afterwards
    """
    import os
    import json
    with open(os.path.join(directory_name, "params.json")) as f:
        d = json.load(f)
    classname = d["type"]
    del d["type"]
    model = globals()[classname](**d)  # remaining dict as named arguments
    if torch.cuda.is_available():
        st = torch.load(os.path.join(directory_name, "model"))
    else:
        st = torch.load(
            os.path.join(directory_name, "model"),
            map_location=torch.device('cpu')  # cannot load to cuda
            )
    model.load_state_dict(st)
    return model


class Identity(torch.nn.Module):
    """
    The identity mapping. Does not decode/encode anything but just returns the
    original input (in appropriate format)

    For debuging purposes
    """
    def __init__(self, dict_size=16384):
        """
        Need dict_size to specify size of output vector.
        """
        super(Identity, self).__init__()
        self.dict_size = dict_size
        # if parameter list is empty, optimizer throws an error.
        self.make_optimizer_shut_up = torch.nn.Linear(1, 1)
        # now the parameter list is not empty anymore.

    def forward(self, input, teacher_forcing):
        """
        Arguments: input, teacher_forcing
        input: should be a LONG tensor of size seq_len x batch_size
        teacher_forcing: whether to use teacher forcing or not.
                         can be boolean or list of length seq_len-1

        Returns: p, pred, code
        p: seq_len x batch_size x dict_size (classification probabilities)
        pred: seq_len x batch_size (prediction of the model, for now it is
              argmax, ie. pred[i, j] = argmax(p[i, j][]), but it can in
              principle also be sampled according to the probability
              distribution given by p)
        code: seq_len x batch_size (code vector) here: identical with input
        """
        seq_len = input.size(0)
        batch_size = input.size(1)
        p = torch.zeros((seq_len, batch_size, self.dict_size),
                        # requires_grad=True,  # won't work, because not leaf
                        device=input.device)
        for i in range(seq_len):
            for j in range(batch_size):
                p[i, j, input[i, j]] = 1.
        # backward() will throw an error if it has nothing to compute.
        p.requires_grad = True
        p2 = p*p  # does nothing, but can compute grads now.
        pred = input
        return p2, pred

    def encode(self, input):
        """
        Returns the code vector of input.
        Arguments: input
        input: should be a LONG tensor of size seq_len x batch_size

        Returns
        code: 1 x batch_size x seq_len (code vector for each seq.)
        (the `1 x' is to be compatible with the other model)
        """
        seq_len, batch_size = input.size()
        return input.t().reshape((1, batch_size, seq_len))

    def save(self, directory_name):
        """
        Saves model to directory directory_name
        """
        import os
        import json
        if not os.path.isdir(directory_name):
            os.mkdir(directory_name)
        torch.save(self.state_dict(), os.path.join(directory_name, "model"))
        # the custom variables are not saved in state dict (only the weights)
        # so save them in dict, along with classname.
        d = {
            "type": type(self).__name__,  # Identity
            "dict_size": self.dict_size,
            }
        with open(os.path.join(directory_name, "params.json"), "w") as f:
            json.dump(d, f, indent=2)


class Autoencoder(torch.nn.Module):
    """
    Wraps up all functionality of the seq2seq translation in one single
    module. It contains:
        - 2 embedding layers to represent the input sequence (encoder, decoder)
        - an encoder (EncoderRNN as in model.encoder)
        - a decoder (DecoderRNN as in model.decoder)
        - a linear and softmax layer to convert the encoded predictions back
          into indices

    This class is as specified in the tutorial
    """
    def __init__(self, encoder_embedding_size=128, decoder_embedding_size=128,
                 hidden_size=128, num_layers=2, reverse=True, dict_size=16384):
        """
        encoder_embedding_size: the dimension of the embedding (encoder)
        decoder_embedding_size: the dimenstion of the embedding (decoder)
        hidden_size: dimension of the hidden vector for a sequence
        num_layers: number of layers of GRU to stack
        reverse: whether to read the input sequence in reverse
        dict_size: number of keys you want to embed

        Note about the dimensions:
        As we want to feed the last hidden vector of the encoder GRU into the
        decoder GRU, the hidden sizes of encoder GRU and decoder GRU must
        agree (ie. hidden_size and num_layers must agree).
        """
        super(Autoencoder, self).__init__()
        # save variables
        self.encoder_embedding_size = encoder_embedding_size
        self.decoder_embedding_size = decoder_embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.reverse = reverse
        self.dict_size = dict_size
        # create embedding layers.
        self.encoder_embedding = torch.nn.Embedding(
            num_embeddings=dict_size,
            embedding_dim=encoder_embedding_size,
            )
        self.decoder_embedding = torch.nn.Embedding(
            num_embeddings=dict_size,
            embedding_dim=decoder_embedding_size,
            )
        # create encoder
        self.encoder = model.encoder.EncoderRNN(
            input_size=encoder_embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers
            )
        # create decoder
        self.decoder = model.decoder.DecoderRNN(
            input_size=decoder_embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers
            )
        # To undo the embedding
        self.linear = torch.nn.Linear(in_features=hidden_size,
                                      out_features=dict_size)
        # dim 0 is sequence offset
        # dim 1 is batch index
        # dim 2 is classification probabilites
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, input, teacher_forcing):
        """
        Arguments: input, teacher_forcing
        input: should be a LONG tensor of size seq_len x batch_size
        teacher_forcing: whether to use teacher forcing or not.
                         can be boolean or list of length seq_len-1

        Returns
        p: seq_len x batch_size x dict_size (classification probabilities,
           actually the logarithm thereof)
        pred: seq_len x batch_size (prediction of the model, for now it is
              argmax, ie. pred[i, j] = argmax(p[i, j][]), but it can in
              principle also be sampled according to the probability
              distribution given by p)
        """
        seq_len = input.size(0)
        batch_size = input.size(1)
        if isinstance(teacher_forcing, bool):
            # turn it into a (constant) list
            teacher_forcing = [teacher_forcing for _ in range(seq_len-1)]

        # get the code vector
        code = self.encode(input)

        # decode
        p_part = []  # partial log probabilities (will cat them in the end)
        pred_part = []  # partial predictions (will cat them in the end)

        # this is how we compute the probabilities p and the prediction pred:
        def forward_and_append(input, h):
            # input is an already embedded part of the sequence and hence of
            # size partial_seq_len x batch_size x embed_size
            # h is the initial hidden vector
            output_i, h_i = self.decoder(input, h)
            # comput log probabilities
            probabilities = self.softmax(self.linear(output_i))
            # compute predictions (argmax of log probabilities)
            with torch.no_grad():
                max = probabilities.topk(1, dim=2)  # max over all tokens
                # max is now a named tuple (values, indices)
                # max.indices has size partial_seq_len x batch_size x 1
                predictions = max.indices.squeeze(dim=2)
            p_part.append(probabilities)
            pred_part.append(predictions)
            return h_i
        # (first) dummy input
        init = self.decoder_embedding(
            torch.zeros(1, batch_size, dtype=torch.long, device=input.device)
            )
        # compute first predictions of batch
        h_i = forward_and_append(init, code)
        i = 1
        while i < seq_len:
            if teacher_forcing[i-1]:
                # we can do multiple teacher-forced predictions at once
                i_next = i+1
                while i_next < seq_len and teacher_forcing[i_next-1]:
                    i_next += 1
                input_i = self.decoder_embedding(input[i-1:i_next-1])
                h_i = forward_and_append(input_i, h_i)
                i = i_next
            else:
                # can do only one non-teacher foced prediction
                # input is previous prediction (last row of last predction)
                input_i = self.decoder_embedding(pred_part[-1][-1:])
                h_i = forward_and_append(input_i, h_i)
                i += 1
        p = torch.cat(p_part)
        pred = torch.cat(pred_part)
        return p, pred

    def encode(self, input):
        """
        Returns the code vector of input.
        Arguments: input
        input: should be a LONG tensor of size seq_len x batch_size

        Returns
        code: num_layers x batch_size x hidden_size (code vector for each seq.)
        """
        seq_len = input.size(0)
        batch_size = input.size(1)
        # embed
        # embed accepts a many-dimensional vector (say dimensions D)
        # and returns a vector of dimensions D x embedding_dim
        # (here D = seq_len x batch_size)
        enc_embed = self.encoder_embedding(input)
        # enc_embed is now of size seq_len x batch_size x enc_embed_size
        # encode
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                          device=enc_embed.device)
        # only interested in h_n
        if self.reverse:
            code = h_0
            for i in range(seq_len-1, -1, -1):
                _, code = self.encoder(enc_embed[i:i+1], code)
        else:
            _, code = self.encoder(enc_embed, h_0)
        return code

    def save(self, directory_name):
        """
        Saves model to directory directory_name
        """
        import os
        import json
        if not os.path.isdir(directory_name):
            os.mkdir(directory_name)
        torch.save(self.state_dict(), os.path.join(directory_name, "model"))
        # the custom variables are not saved in state dict (only the weights)
        # so save them in dict, along with classname.
        d = {
            "type": type(self).__name__,  # Autoencoder
            "encoder_embedding_size": self.encoder_embedding_size,
            "decoder_embedding_size": self.decoder_embedding_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "reverse": self.reverse,
            "dict_size": self.dict_size,
            }
        with open(os.path.join(directory_name, "params.json"), "w") as f:
            json.dump(d, f, indent=2)


class Autoencoder_SE(torch.nn.Module):
    """
    Wraps up all functionality of the seq2seq translation in one single
    module. It contains:
        - an embedding layer to represent the input sequence
        - an encoder (EncoderRNN as in model.encoder)
        - a decoder (DecoderRNN as in model.decoder)
        - a linear and softmax layer to convert the encoded predictions back
          into indices

    This class differs from the turorial in the following ways:
        - It Shares Embedding between encoder and decoder
    """
    def __init__(self, embedding_size=128, hidden_size=128, num_layers=2,
                 reverse=True, dict_size=16384):
        """
        embedding_size: the dimension of the embedding
        hidden_size: dimension of the hidden vector for a sequence
        num_layers: number of layers of GRU to stack
        reverse: whether to read the input sequence in reverse
        dict_size: number of keys you want to embed

        Note about the dimensions:
        Since there is only one embedding, the input size of the encoder and
        decoder GRU must coincide with the dimension of the embedding. As we
        want to feed the last hidden vector of the encoder GRU into the
        decoder GRU, the hidden sizes of encoder GRU and decoder GRU must also
        agree.
        """
        super(Autoencoder_SE, self).__init__()
        # save variables
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.reverse = reverse
        self.dict_size = dict_size
        # create embedding layer.
        self.embedding = torch.nn.Embedding(num_embeddings=dict_size,
                                            embedding_dim=embedding_size)
        # create encoder
        self.encoder = model.encoder.EncoderRNN(input_size=embedding_size,
                                                hidden_size=hidden_size,
                                                num_layers=num_layers)
        # create decoder
        self.decoder = model.decoder.DecoderRNN(input_size=embedding_size,
                                                hidden_size=hidden_size,
                                                num_layers=num_layers)
        # To undo the embedding
        self.linear = torch.nn.Linear(in_features=hidden_size,
                                      out_features=dict_size)
        # dim 0 is sequence offset
        # dim 1 is batch index
        # dim 2 is classification probabilites
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, input, teacher_forcing):
        """
        Arguments: input, teacher_forcing
        input: should be a LONG tensor of size seq_len x batch_size
        teacher_forcing: whether to use teacher forcing or not.
                         can be boolean or list of length seq_len-1

        Returns:
        p: seq_len x batch_size x dict_size (classification probabilities,
           actually the logarithm thereof)
        pred: seq_len x batch_size (prediction of the model, for now it is
              argmax, ie. pred[i, j] = argmax(p[i, j][]), but it can in
              principle also be sampled according to the probability
              distribution given by p)
        """
        seq_len = input.size(0)
        batch_size = input.size(1)
        if isinstance(teacher_forcing, bool):
            # turn it into a (constant) list
            teacher_forcing = [teacher_forcing for _ in range(seq_len-1)]
        # embed
        # embed accepts a many-dimensional vector (say dimensions D)
        # and returns a vector of dimensions D x embedding_dim
        # (here D = seq_len x batch_size)
        embedded = self.embedding(input)
        # embedded is now of size seq_len x batch_size x embed_size

        # encode
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                          device=embedded.device)
        # only interested in h_n
        if self.reverse:
            code = h_0
            for i in range(seq_len-1, -1, -1):
                _, code = self.encoder(embedded[i:i+1], code)
        else:
            _, code = self.encoder(embedded, h_0)

        # decode
        p_part = []  # partial log probabilities (will cat them in the end)
        pred_part = []  # partial predictions (will cat them in the end)

        # this is how we compute the probabilities p and the prediction pred:
        def forward_and_append(input, h):
            # input is an already embedded part of the sequence and hence of
            # size partial_seq_len x batch_size x embed_size
            # h is the initial hidden vector
            output_i, h_i = self.decoder(input, h)
            # compute log probabilities
            probabilities = self.softmax(self.linear(output_i))
            # compute predictions (argmax of log probabilities)
            with torch.no_grad():
                max = probabilities.topk(1, dim=2)  # max over all tokens
                # max is now a named tuple (values, indices)
                # max.indices has size partial_seq_len x batch_size x 1
                predictions = max.indices.squeeze(dim=2)
            p_part.append(probabilities)
            pred_part.append(predictions)
            return h_i  # last hidden vector
        # (first) dummy input
        init = self.embedding(
            torch.zeros(1, batch_size, dtype=torch.long, device=input.device)
            )
        # init is now of size 1 x batch_size x embed_size
        # compute first predictions of batch
        h_i = forward_and_append(init, code)
        i = 1
        while i < seq_len:
            if teacher_forcing[i-1]:
                # we can do multiple teacher-forced predictions at once
                i_next = i+1
                while i_next < seq_len and teacher_forcing[i_next-1]:
                    i_next += 1
                h_i = forward_and_append(embedded[i-1:i_next-1], h_i)
                i = i_next
            else:
                # can do only one non-teacher foced prediction
                # input is previous prediction (last row of last predction)
                input_i = self.embedding(pred_part[-1][-1:])
                h_i = forward_and_append(input_i, h_i)
                i += 1
        p = torch.cat(p_part)
        pred = torch.cat(pred_part)
        return p, pred

    def encode(self, input):
        """
        Returns the code vector of input.
        Arguments: input
        input: should be a LONG tensor of size seq_len x batch_size

        Returns
        code: num_layers x batch_size x hidden_size (code vector for each seq.)
        """
        seq_len = input.size(0)
        batch_size = input.size(1)
        embedded = self.embedding(input)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                          device=embedded.device)
        if self.reverse:
            code = h_0
            for i in range(seq_len-1, -1, -1):
                _, code = self.encoder(embedded[i:i+1], code)
        else:
            _, code = self.encoder(embedded, h_0)
        return code

    def save(self, directory_name):
        """
        Saves model to directory directory_name
        """
        import os
        import json
        if not os.path.isdir(directory_name):
            os.mkdir(directory_name)
        torch.save(self.state_dict(), os.path.join(directory_name, "model"))
        # the custom variables are not saved in state dict (only the weights)
        # so save them in dict, along with classname.
        d = {
            "type": type(self).__name__,  # Autoencoder_SE
            "embedding_size": self.embedding_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "reverse": self.reverse,
            "dict_size": self.dict_size,
            }
        with open(os.path.join(directory_name, "params.json"), "w") as f:
            json.dump(d, f, indent=2)
