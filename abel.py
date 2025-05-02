import torch
import torch.nn as nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
import time
import pandas as pd



#hyperparameters
batch_size = 256 #no. of independent sequences processed in parallel
block_size = 128 #blocksize is the length of chunk fed in transformer for training. completely randon, and each chunk is charcters in order. here i have kept it 32
max_iters = 10000
eval_interval = 1000
learning_rate = 1e-4
device =  'cpu'
eval_iters = 200
n_embed = 384  #no. of embedding dimensions
"""
important to ensure (batch_size * block_size) = n_ebed,
as this then used in matrix mul in feed fwd
"""

n_head = 6
n_layer = 6
dropout = 0.4
#----------------------

torch.manual_seed(1337)

#----------------------



df = pd.read_csv("hf://datasets/vishnupriyavr/spotify-million-song-dataset/spotify_millsongdata.csv")
text = df['text']
#here are the unique chars that are in the text
text = '\n'.join(text)
chars = sorted(list(set(text)))
vocab_size = len(chars)

#create a mapping from char to int and vice-versa
stoi = { ch:i for i, ch in enumerate(chars)} #this is a dict, of format{chr: index}
itos = {i : ch for i, ch in enumerate(chars)} #{index : chr}
encode = lambda s: [stoi[c] for c in s] #takes a str , output list of int
decode = lambda l : ''.join([itos[i] for i in l]) #takes a list of int, outputs a str

#Train and Test splits
data = torch.tensor(encode(text), dtype = torch.long) #TURNS THE ENCODED DATA INTO A TENSOR(MULTI DIMENSIONED ARRAY)
n = int(0.9 * len(data)) #90% is training
train_data = data[:n]
val_data = data[n:]

#data loading
def get_batch(split):
    #generate a small batch of data inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, )) # randomly generates a certain amount of numbers between 0 and (len(data)-block size). the number of rnos depends on batchsize, so 4 here
    x = torch.stack([data[i : i + block_size] for i in ix]) #creates a batch of input sequences tensor from data, and stacks them
    y = torch.stack([data[i+1 : i+block_size+1] for i in ix]) #same, but offset by one
    x , y = x.to(device), y.to(device)  #moves tensors to gpu
    return x,y

@torch.no_grad()  #disables gradient computation during the function execution. This is useful for inference and evaluation, where we don’t need to update weights.
def estimate_loss():
    #averages out loss in multiple batches
    out = {}  # Dictionary to store average losses for train and validation sets
    model.eval() # Set the model to evaluation mode
    for split in ['train', 'val']:   # Loop through training and validation datasets
        losses = torch.zeros(eval_iters)  # Tensor to store loss values
        for k in range(eval_iters):
            X, Y = get_batch(split)    # Get a batch of training/validation data
            logits, loss = model(X, Y)  # Forward pass (compute predictions and loss)
            losses[k] = loss.item()  # Store the loss value
        out[split] = losses.mean() #compute avg loss
    model.train()  #switch model back to training mode
    return out

class Head(nn.Module):
    """One head of self attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias = False)   #Linear layer to generate Key (K)
        self.query = nn.Linear(n_embed, head_size, bias = False)   #Linear layer to generate query(q)
        self.value = nn.Linear(n_embed, head_size, bias=False)   #Linear layer to generate value(v)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  #This creates and stores a lower triangular matrix (tril) inside the model, ensuring it persists with the model's state but doesn’t update during training.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        #compute attention scores (affinities)
        wei = q @ k.transpose(-2, -1) * C**-0.5  #(B, T, 16) @ (B, 16, T) -----> (B,T, T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) #(B,T,T) ; decoder block, doesnt allow communication from the past
        wei = F.softmax(wei, dim = -1)  #(B,T,T)
        wei = self.dropout(wei)  #randomly sets some values in wei to zero during training
        #perform weighted aggregation of the vallues
        v = self.value(x)
        out = wei @ v  #(B,T,T) @ (B, T, C) ----> (B, T, C )
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        #ensure each attention head is part of the model
        self.proj = nn.Linear(n_embed, n_embed) #use to project input feature to a diff space, wile retaining dimesionality. makes model flexible (input, output)
        #it basically allows model to refine the results and add nes=cessary transformations
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        #each head processes x independently, and all the outputs ac=re concatenated along the feature dimesion dim = 1
        #output shape = (B, T, num_headsXhead_size)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """a simple linaer layer followed by a non linearity
       IT ALLOWS TOKENS TO NOT ONLY INTERACT, BUT THINK, IE, HOW THEY INTERACT WITH EACH OTHER
    """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential( #sequential allows to define a sequence of layers applied to the input
            nn.Linear(n_embed, 4 * n_embed),  #fully connected linear layer, that takes an input of size n_embed and prods output of n_embed
            nn.GELU(),  #gives non linearity to the func
            nn.Linear(4 * n_embed,n_embed),     #projection layer/intermediate layer, to enchance model  #converts output to n_embed
            nn.Dropout(dropout) #Dropout is a regularization technique used in neural networks to prevent overfitting. It randomly sets some neurons to zero during training, forcing the model to learn more robust features.
        )  #inner layer is 4 times, and 2 times the outer layer(this is mathematical )

    def forward(self, x):
        return self.net(x)  #sends data through the layers in self.net


class Block(nn.Module):
    """Transformer Block : communication followed ny computation
        combines multi head attention and feedforward. basic of transformer arch.

    """
    def __init__(self, n_embed, n_head):
        #n_embed : embedding dimension, n_head: the no. of heads we would like
        super().__init__()
        head_size = n_embed// n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        #layer normalization to stabilise training, twice
        self.ln1 = nn.LayerNorm(n_embed) # before self attention
        self.ln2 = nn.LayerNorm(n_embed)  #before feedforward nw


    def forward(self,x):
        x = x + self.sa(self.ln1(x))  #COMMUNICATION - self-attention to allow the model to focus on different parts of the input sequence.
        x = x + self.ffwd(self.ln2(x))  #COMPUTATION - a feed-forward network is applied to each position in the sequence independently.

        #x + enables residual block, who contribute as optimisers in the propogation path,
        # and allows us to fork off, do some comm/comp, and come back
        return x


#super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        #creates a table of nXm., which has a thin abstraction(nn.embedding). each input plucks out a row corresponding to its num.
        #ex, 43 takes row 43, 54 takes row 54(keep in mind, this is on a small set on random chars)
        '''
        nn.Embedding(vocab_size, vocab_size) creates a lookup table (embedding matrix).
        This means that:
            Every token (word/character) has a unique vector representation (embedding).
            Since both input and output sizes are vocab_size, each token maps directly to a vector of size vocab_size.
            This table is randomly initialized and will be learned during training.

             embedding dimensions refer to the size of the vector used to represent each discrete token
             Embedding is the process of converting discrete variables (like words, characters, etc.) into continuous
             vectors in a higher-dimensional space. This helps neural networks process and learn from textual data,
             which can be difficult to work with in its raw form

        '''
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head = n_head) for _ in range(n_layer)])  #stack of transfermer blocks, via which data passes
        self.ln_f = nn.LayerNorm(n_embed) #final layer normalization
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        #logit is score for next char in sequennce
        #above rows are arranged in bxtxc
        tok_emb = self.token_embedding_table(idx)  #BXTXC
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  #T,c,,  transformers don't have recurrence (like RNNs), so they need positional embeddings to capture word order.
        x = tok_emb + pos_emb
        x = self.blocks(x)  #(B,T, C)
        logits = self.lm_head(x) #(B, T, vocabsize)
        '''
        This retrieves the embeddings (vectors) for each token index in idx from self.token_embedding_table.
        '''
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) #pytorch devs, in their infinite knowledge, asked bXCXt for loss function, flattening the batch and time dimensions into a single dimension, while keeping the C dimension the same.
            targets = targets.view(B*T) #latten it to (B*T,) so it matches the new logits shape.
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        #idx is bXT array of indice in current context
        for _ in range(max_new_tokens):
            #crop idx to the last block_size token, to prevent the embedding table from ceashing
            idx_cond = idx[:, -block_size:]
            #get the predictions
            logits, loss = self(idx_cond)
            #focus only on the last TIME step
            logits = logits[:, -1, :] #becomes (B, C)
            """logits[:, -1, :] means "take the last word in each sentence"
                Shape becomes (B, C) = (3, 10000) → 3 rows (one per sentence), each with 10,000 possible next words."""
            #apply softmax to get probablities
            probs = F.softmax(logits, dim = -1) #(B, C)
            #sample fromm the distribution
            idx_next = torch.multinomial(probs, num_samples = 1) #(B, 1)   #Instead of always picking the most probable token (which can lead to repetitive text), it samples to introduce randomness.
            #append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim = 1)  #(B, T+1)
        return idx

if __name__ == "__main__":
    device_avail = torch.cuda.is_available()

    print(f"{torch.cuda.get_device_name()} IS {device_avail}")

    start_time = time.time()
    torch.cuda.empty_cache()

    model = BigramLanguageModel()

    m = model.to(device) #movexs model param to gpu

    #create a pytorch optimizer, to optimise weights during training, here we use AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay=1e-4)  #Weight decay adds a penalty to large weights, forcing them to stay smaller and preventing the model from memorizing training data.



    tloss, vloss = [],[]
    #training loop
    for iter in range(max_iters):

        print(f"\r Iteration No.: {iter} --> ", end = '', flush= True)

        #every once in a while, eval the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            tloss.append(losses['train'])
            vloss.append(losses['val'])

        #sample a batch of data
        xb, yb = get_batch('train')
        #eval the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none= True)  # Clears old gradients
        loss.backward()   # Computes gradients (backpropagation)
        optimizer.step()   # Updates model weights


    #generate from the mode

    model_sve = torch.save(model.state_dict(), 'ABEL_.pth')

    print("\n\n")

    context = torch.zeros((1,1), dtype = torch.long, device = device)
    print(decode(m.generate(context, max_new_tokens= 1000)[0].tolist())) #[0] extracts the first batch, which is then converted in readable txt

    end_time = time.time()
    elasped_time = (end_time - start_time)/3600
    print(f"\n\nProgram Run Time : {elasped_time:.4f}")

    plt.xlabel('Training loss')
    plt.ylabel('Validation loss')
    plt.title('Training Loss V/S Validation Loss')

    plt.plot(tloss)
    plt.plot(vloss)


    plt.show()

