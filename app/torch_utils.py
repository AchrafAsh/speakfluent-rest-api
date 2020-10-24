import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio

class MVPModel(nn.Module):
    def __init__(self, batch_size: int):
        super(MVPModel, self).__init__()
        self.rnn = nn.LSTM(input_size=128+28, hidden_size=20,
                           num_layers=2, bidirectional=True)
        self.fc = nn.Linear(40, 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, specgram, label, initial_states):
        # initial states
        h0, c0 = initial_states

        x = torch.cat((label, specgram), 2)

        output, (hn, cn) = self.rnn(x, (h0, c0))
        output = self.fc(output)
        return self.softmax(output)


# load model
model = MVPModel(batch_size=4)
model.load_state_dict(torch.load("speakfluent_mvp.pth"))
model.eval()

# data preprocessing
class TextTransform:
    """ Maps characters to integers and vice versa """
    def __init__(self):
        char_map_str = """
            ' 0
            <SPACE> 1
            a 2
            b 3
            c 4
            d 5
            e 6
            f 7
            g 8
            h 9
            i 10
            j 11
            k 12
            l 13
            m 14
            n 15
            o 16
            p 17
            q 18
            r 19
            s 20
            t 21
            u 22
            v 23
            w 24
            x 25
            y 26
            z 27
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence
    
    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('', ' ')
    
    def tensor_to_text(self, tensor):
        word = tensor.transpose(1,0)
        output = ['']*len(tensor)
        for i in range(len(tensor)):
            for j in range(28):
                if word[j][i] == 1: output[i] = self.index_map[j]
        print(output)
        return ''.join(output)

    def one_hot_enc(self, word):
        """ Returns a sequence of ones and zeros, result of one hot encoding """
        word = self.text_to_int(word)
        word = Variable(torch.tensor(word))
        word = torch.nn.functional.one_hot(word, len(self.index_map))
        return word.transpose(0, 1)

text_transform = TextTransform()

def preprocessing(audio_filename: str, word: str):
    waveform, _ = torchaudio.load(audio_filename)

    label = text_transform.one_hot_enc(word).transpose(1,0)

    # spectrogram
    specgram = torchaudio.transforms.MelSpectrogram()(waveform)
    specgram = F.interpolate(specgram, size=len(word), mode="nearest").transpose(1, 2)

    return specgram.unsqueeze(0), label.unsqueeze(0)

def get_prediction(spectrogram, label):
    spectrogram = spectrogram.transpose(0,1)[:,:,0,:]
    _,word_size,_ = label.size()

    # initial states
    h0 = torch.ones(4, word_size, 20)
    c0 = torch.ones(4, word_size, 20)

    return model(spectrogram, label, (h0, c0))[:,:,0]
