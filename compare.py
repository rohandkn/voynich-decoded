import torch
import numpy as np
from sklearn.metrics import roc_auc_score


# right now dataloading and model inputs are simplified
# may have to change to fit the formatting of our test data

# Generates embeddings for a given input
def get_embeddings_in_device(model, text, device=torch.device('cpu')):
    model.eval()
    text = text.to(device)
    embedding = model.forward(text)
    return embedding

def compare_text(model, text1, text2, similarityFn, device=torch.device('cpu')):

    text1 = text1.to(device)
    text2 = text2.to(device)
    embedding1 = get_embeddings_in_device(model, text1, device)
    embedding2 = get_embeddings_in_device(model, text2, device)
    similarity = similarityFn(embedding1, embedding2)

    return similarity.cpu().detach().numpy()


def get_similarity_scores(model, dataloader, similarityFn, device=torch.device('cpu')):
    model.eval()
    similarityScores = []

    for i, (text1, text2) in enumerate(dataloader):
        similarity = compare_text(model, text1, text2, similarityFn, device)
        similarityScores.append(similarity)

    return np.array(similarityScores)


def get_roc_auc_score(model, dataloader, similarityFn, device=torch.device('cpu')):
    similarityScores = get_similarity_scores(model, dataloader, similarityFn, device)

    return roc_auc_score(dataloader.dataset.labels, similarityScores)

