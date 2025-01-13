class LSTMConfig:
    def __init__(self):
<<<<<<< HEAD
        self.x_input_size = 7
=======
        self.x_input_size = 6
>>>>>>> 6a91b5d423c756b82df34fa1d19ee44af9e1ac77
        self.x_output_size = 3

        self.P_input_size = 18
        self.P_output_size = 9

<<<<<<< HEAD
        self.hidden_size = 64
=======
        self.hidden_size = 128
>>>>>>> 6a91b5d423c756b82df34fa1d19ee44af9e1ac77
        self.num_layers = 4
        self.dropout = 0.0

    def getLSTMConfig(self):
        return self.x_input_size, self.x_output_size, self.hidden_size, self.num_layers, self.dropout, self.P_input_size, self.P_output_size
