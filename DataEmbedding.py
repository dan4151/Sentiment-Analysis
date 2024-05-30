import re
from gensim import downloader
import torch
import numpy as np
from torch.utils.data import DataLoader


class DataEmbedding:
    def __init__(self, vec_size):

        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.vec_size = vec_size
        self.embedding_path = "glove-twitter-100"
        self.embedding_model = downloader.load(self.embedding_path)
        self.unknown_word = torch.rand(self.vec_size, requires_grad=True)


    def get_sen_embedding_from_path(self, df, is_tagged, test_or_train):
        X = []
        tags = []
        sen_lens = []
        mapping = {'happiness': 2, 'neutral': 1, 'sadness': 0}
        df['emotion'] = df['emotion'].replace(mapping)
        for index, row in df.iterrows():
            if is_tagged:
                tag = row['emotion']
            sentence = row['content']
            pattern = r'[^a-zA-Z\s]'
            sentence = re.sub(pattern, ' ', sentence)
            words = sentence.split()
            sentence_vec = []
            for word in words:
                word = word.strip()
                word = word.lower()
                if word in self.embedding_model.key_to_index:
                    sentence_vec.append(torch.tensor(self.embedding_model[word]))
            if len(sentence_vec) == 0:
                continue
            sen_lens.append(len(sentence_vec))
            X.append(sentence_vec)
            tags.append(tag)

        padding_tensor = torch.zeros(self.vec_size)
        max_length = max(len(sublist) for sublist in X)
        X = [sublist + [padding_tensor] * (max_length - len(sublist)) for sublist in X]
        X = [torch.stack(vec) for vec in X]
        if is_tagged:
            y = [torch.tensor(a) for a in tags]
            y = torch.tensor(y)
        data_set = [*zip(X, y, sen_lens)]
        data_loader = DataLoader(data_set, batch_size=32, shuffle=True)
        if is_tagged:
            return data_loader, tags



