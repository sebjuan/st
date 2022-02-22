from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd


def extract_best_indices(m, topk):
    """
    Use sum of the cosine distance over all tokens.
    m (np.array): cos matrix of shape (nb_in_tokens, nb_dict_tokens)
    topk (int): number of indices to return (from high to lowest in order)
    """
    # return the sum on all tokens of cosinus for each sentence

    print(f"m is {m}")
    if len(m.shape) > 1:
        cos_sim = np.mean(m, axis=0)
    else:
        cos_sim = m
    index = np.argsort(cos_sim)[::-1]  # from highest idx to smallest score

    scores = np.sort(cos_sim)[::-1]
    mask = np.ones(len(cos_sim))

    # print(f"mask is {mask}")
    # mask = np.logical_or(cos_sim[index] != 0, mask)  # eliminate 0 cosine distance
    # print(f"mask is {mask}")
    # best_index = index[mask][:topk]

    best_index = index[:topk]
    best_scores = scores[:topk]
    print(f"best index is {best_index}")
    print(f"scores are : {scores}")
    return best_index, best_scores


BERT_BATCH_SIZE = 40


# MODEL_NAME = 'sentence-transformers/paraphrase-MiniLM-L6-v2'

class BertModel:
    def __init__(self, model_name, device=-1, small_memory=True, batch_size=BERT_BATCH_SIZE):
        self.model_name = model_name
        self._set_device(device)
        self.small_device = 'cpu' if small_memory else self.device
        self.batch_size = batch_size
        self.load_pretrained_model()

    def _set_device(self, device):
        if device == -1 or device == 'cpu':
            self.device = 'cpu'
        elif device == 'cuda' or device == 'gpu':
            self.device = 'cuda'
        elif isinstance(device, int) or isinstance(device, float):
            self.device = 'cuda'
        else:  # default
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

    def load_pretrained_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        device = -1 if self.device == 'cpu' else 0
        self.pipeline = pipeline('feature-extraction',
                                 model=self.model, tokenizer=self.tokenizer, device=device)

    def embed(self, data):
        """ Create the embedded matrice from original sentences """
        nb_batchs = 1 if (len(data) < self.batch_size) else len(
            data) // self.batch_size
        batchs = np.array_split(data, nb_batchs)
        mean_pooled = []
        for batch in tqdm(batchs, total=len(batchs), desc='Training...'):
            mean_pooled.append(self.transform(batch))
            # print(f"mean_pooled is {mean_pooled}")
        mean_pooled_tensor = torch.tensor(
            len(data), dtype=float).to(self.small_device)
        mean_pooled = torch.cat(mean_pooled, out=mean_pooled_tensor)
        # print(f"mean_pooled = embeded_mat is {mean_pooled}")
        self.embed_mat = mean_pooled

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        # print(f"token_embeddings is {token_embeddings}")
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                  min=1e-9)

    def transform(self, data):
        if 'str' in data.__class__.__name__:
            data = [data]
        data = list(data)
        token_dict = self.tokenizer(
            data,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt")
        token_dict = self.to(token_dict, self.device)
        with torch.no_grad():
            token_embed = self.model(**token_dict)
        # each of the 512 token has a 768 or 384-d vector depends on model)
        attention_mask = token_dict['attention_mask']
        # average pooling of masked embeddings
        mean_pooled = self.mean_pooling(
            token_embed, attention_mask)
        mean_pooled = mean_pooled.to(self.small_device)
        return mean_pooled

    def to(self, data: dict, device: str):
        """Send all values to device by calling v.to(device)"""
        data = {k: v.to(device) for k, v in data.items()}
        return data

    def predict(self, in_sentence, topk=4):
        input_vec = self.transform(in_sentence)
        mat = cosine_similarity(input_vec, self.embed_mat)
        # best cos sim for each token independantly
        best_index, best_score = extract_best_indices(mat, topk=topk)
        return best_index, best_score


MODEL_NAME = 'sentence-transformers/all-mpnet-base-v1'
BERT_BATCH_SIZE = 4
bert_model = BertModel(model_name=MODEL_NAME, batch_size=BERT_BATCH_SIZE)

matrice = torch.load('mon_premier_torch.pt')
df = pd.read_csv("films.txt", header='infer')
bert_model.embed_mat = matrice


def get_random_plot():
    i = np.random.randint(len(df.index))
    return df.loc[i, 'RAW_PLOT']
