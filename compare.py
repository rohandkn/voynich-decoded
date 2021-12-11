import torch
import numpy as np
from torch import nn
from sklearn.metrics import roc_auc_score

from generate_test_data import load_test_data


# Generates embeddings for a given input
# assumes model is loaded into device
# if model is None, a random torch tensor is returned as the embedding
def get_embeddings_in_device_basic(model, text, device=torch.device('cpu')):
    if (model != None):
        model.eval()

    b_text = tuple(t.to(device) for t in text)
    b_input_ids, b_input_mask = b_text

    # # this is for now, can change to a different embedding representation
    if (model != None):
        embedding = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        return embedding

    # for testing purposes
    return torch.randn((len(b_input_ids), 10))

def get_embeddings_in_device(model, text, device=torch.device('cpu')):
    pass

def compare_text(model, text1, text2, similarityFn, device=torch.device('cpu')):

    embedding1 = get_embeddings_in_device_basic(model, text1, device)
    embedding2 = get_embeddings_in_device_basic(model, text2, device)
    similarity = similarityFn(embedding1, embedding2)

    return similarity.cpu().detach().numpy()


def get_similarity_scores(model, dataloader, similarityFn, device=torch.device('cpu')):
    similarityScores = []

    # similarity scores are generated in batches
    for i, (inputs, masks, _) in enumerate(dataloader):
        b_text1 = (inputs[:,0], masks[:,0])
        b_text2 = (inputs[:,1], masks[:,1])
        similarity = compare_text(model, b_text1, b_text2, similarityFn, device)
        similarityScores.append(similarity)

    return np.concatenate(similarityScores)


def get_labels(dataloader):
    labels = []
    for i, (_, _, label) in enumerate(dataloader):
        label = label.cpu().detach().numpy()
        labels.append(label)

    return np.concatenate(labels)


def get_roc_auc_score(model, dataloader, similarityFn, device=torch.device('cpu')):
    similarityScores = get_similarity_scores(model, dataloader, similarityFn, device)

    labels = get_labels(dataloader)
    return roc_auc_score(labels, similarityScores)


#### for testing
if __name__ == "__main__":
    dataset = load_test_data("test_inputs.pt", "test_masks.pt", "test_labels.pt")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    model = None
    similarityFn = nn.CosineSimilarity()
    print(get_roc_auc_score(model, dataloader, similarityFn))
