import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.utils.data # used to load datasets
from torchvision import datasets # used to load mnist
from torchvision import transforms # used to Convert a numpy.ndarray to a tensor
from torchvision.utils import save_image

from tqdm import tqdm

seed=1234
batch_size=200
log_interval=50
epochs=50

torch.manual_seed(seed)
device = torch.device("cpu")

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/mnist', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/mnist', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        n_input = 1
        self.n_hidden = 32
        self.n_layers = 1
        self.rnn = nn.RNN(n_input, self.n_hidden, self.n_layers, nonlinearity="relu")
        self.rnn_type = "RNN_RELU" # required
        self.decoder = nn.Linear(self.n_hidden, n_input) # reconstruction

    def forward(self, x, hidden):
        output, hidden = self.rnn(x, hidden)
        decoded = F.sigmoid(self.decoder(output))
        return decoded, hidden


    def init_hidden(self, bsz): # required
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.n_layers, bsz, self.n_hidden),
                    weight.new_zeros(self.n_layers, bsz, self.n_hidden))
        else:
            return weight.new_zeros(self.n_layers, bsz, self.n_hidden)

model = RNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Reconstruction  summed over all elements and batch
def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), size_average=False)
    return BCE

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def train(epoch):
    model.train() # set mode to training
    train_loss = 0
    for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc="Train")):
        data = data.to(device)

        hidden = model.init_hidden(data.shape[0]) # batch size varies based on iteration
        recon_batch, hidden = model(data.view(784, -1, 1), hidden)

        optimizer.zero_grad() # clear gradients each iteration before backprop

        loss = loss_function(recon_batch, data)
        train_loss += loss.item() # add value of tensor as Python number
        
        # note that quite a bit is implicit below:
        loss.backward()       # in comp graph, accumulate gradients at 
                              # relevent differentiable points
        optimizer.step()      # apply gradients to parameters
  
    print('Average loss: {:.4f}'.format(train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval() # set mode to evaluation
    test_loss = 0
    with torch.no_grad(): # turn of gradient calculation in forward pass
                          # seems that this is automatic
        for i, (data, _) in enumerate(tqdm(test_loader, desc="Test")):
            data = data.to(device)
            hidden = model.init_hidden(data.shape[0]) # batch size varies based on iteration
            recon_batch, hidden = model(data, hidden)
            test_loss += loss_function(recon_batch, data).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/rnn_reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('Test set loss: {:.4f}'.format(test_loss))

for epoch in range(1, epochs + 1):
    print("Epoch %d" % epoch)
    train(epoch)
    test(epoch)