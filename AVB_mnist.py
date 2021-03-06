from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/mnist', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/mnist', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

INPUT_SIZE = 784
NCLASSES = 10
HIDDEN_SIZE = 64
HIDDENL_SIZE = 256

class AVB(nn.Module):
    def __init__(self):
        super(AVB, self).__init__()

        # Encoder
        self.enc_net = nn.Sequential(nn.Linear(INPUT_SIZE, HIDDEN_SIZE), nn.ELU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.ELU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.ELU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.ELU())
        self.encode = lambda x, eps: self.enc_net(torch.cat((x, eps), -1))

        # Decoder
        self.dec_net = nn.Sequential(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.ELU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.ELU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.ELU(),
            nn.Linear(HIDDEN_SIZE, NCLASSES))
        self.decode = lambda z: self.dec_net(z)

        # Discriminator
        self.dec1 = nn.Sequential(nn.Linear(HIDDEN_SIZE, HIDDENL_SIZE), nn.ELU())
        self.dec1_dnets = []
        for i in range(5):
            self.dec1_dnets[i] = nn.Sequential(
                nn.Linear(HIDDENL_SIZE, HIDDENL_SIZE), nn.ReLU(),
                nn.Linear(HIDDENL_SIZE, HIDDENL_SIZE))
        self.dec_out = nn.Linear(HIDDENL_SIZE, 1)

    def discriminator(x, z):
        input = torch.cat((x, z), -1)
        h = self.dec1(input)
        for i in range(5):
            h += self.dec1_dnets[i](h)
            h = F.ELU(h)
        print("check form of z")
        import ipdb; ipdb.set_trace()
        o = self.dec_out(h) + torch.sum(z.pow(2).view(-1))
        return o


    def forward(self, x, eps):
        phi = self.phi(x)
        # mu, logvar = self.encode(x.view(-1, 784))
        # z = self.reparameterize(mu, logvar)
        # return self.decode(z), mu, logvar


model = AVB().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc="Train")):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('Average loss: {:.4f}'.format(train_loss / len(train_loader.dataset)))

    #     if batch_idx % args.log_interval == 0:
    #         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #             epoch, batch_idx * len(data), len(train_loader.dataset),
    #             100. * batch_idx / len(train_loader),
    #             loss.item() / len(data)))

    # print('====> Epoch: {} Average loss: {:.4f}'.format(
    #       epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(tqdm(test_loader, desc="Test")):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'oresults/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    print()
    print("Epoch", epoch)
    train(epoch)
    test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, 20).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'oresults/sample_' + str(epoch) + '.png')