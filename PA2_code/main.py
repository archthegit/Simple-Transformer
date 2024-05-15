import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import Decoder
import constants as c


def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :c.block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, c.block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(c.device), Y.to(c.device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(c.device), Y.to(c.device)
        logits, loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def main():

    print("Loading data and creating tokenizer ...")

    texts = load_texts('../speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "../speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=c.batch_size,collate_fn=collate_batch,shuffle=True)

    inputfile = "/Users/Archana/Downloads/CSE256_PA2_SP24/speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  c.block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=c.batch_size, shuffle=True)


    ########################
    #       ENCODER        #
    ######################## 

     # for the classification  task, you will train for a fixed number of epochs like this:
    for epoch in range(c.epochs_CLS):
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(c.device), yb.to(c.device)
            # CLS training code here


    ########################
    #       DECODER        #
    ######################## 

    m = Decoder(vocab_size=tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(m.parameters(), lr=c.learning_rate)

    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= c.max_iters:
            break
        if i % c.eval_interval == 0:
            losses = compute_perplexity(m, train_LM_loader, eval_iters=c.eval_iters)
            print(f"step {iter}: train loss {losses:.4f}")

        xb, yb = xb.to(c.device), yb.to(c.device)
        
        # LM training code here
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    context = torch.zeros((1,1), dtype=torch.long, device=c.device)
    print(tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist()))       

    
 


if __name__ == "__main__":
    main()


