import torch


class DecoderRNN(torch.nn.Module):
    """
    The decoder is a simple 2-layered GRU, but whose input is first passed
    through a ReLU layer.
    """
    def __init__(self, input_size=256, hidden_size=128,
                 num_layers=2):
        """
        input_size: dimension of the input size to GRU
        hidden_size: dimension of the hidden vector for a sequence
        num_layers: number of layers of GRU to stack
        """
        super(DecoderRNN, self).__init__()
        # save variables
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # ReLU
        self.relu = torch.nn.ReLU()
        # decoder GRU
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers)

    def forward(self, input, h_0):
        """
        Arguments: input, h_0
        input: tensor of size  seq_len x batch_size x input_size
        h_0: tensor of size num_layers x batch_size x hidden_size
        Returns: output, h_n
        output: seq_len x batch_size x dict_size (= predictions)
        h_n: num_layers x batch_size x hidden_size (should be ignored)
        (Note: the n in h_n stands for seq_len)
        """
        relued = self.relu(input)
        output, h_n = self.gru(relued, h_0)
        return output, h_n
