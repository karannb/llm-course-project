'''
Training Module.
1.) Saves checkpoints
2.) NO early stopping, on purpose.
'''

from transformer_base import Decoder
from transformer_xl import DecoderMemLM

import matplotlib.pyplot as plt

import torch
import torch.optim as optim

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-4
CONTEXT_LENGTH = 65
COUNTS = 1

class TweetSentiment:

    def __init__(self, path_to_dataset : str = "data/data-16.pt", context_length : int = 65, 
                 batch_size : int = 64, training_xl : bool = False):

        self.x = torch.load(path_to_dataset)

        print(f"{len(self.x)} number of total tokens!", flush=True)

        # Split into train and val
        self.x_val = self.x[int(0.9*len(self.x)):]
        self.x = self.x[:int(0.9*len(self.x))]

        self.context_length = context_length
        self.batch_size = batch_size
        self.training_xl = training_xl
        self.counts = COUNTS # does not increment
        self.counter = 0 # increments!
        self.inds = None
        self.first_it = True # workaround, #FIXME

    def get_batch(self, val = False):

        if self.training_xl and (self.first_it or self.counter == self.counts):
            inds = torch.randint(0, len(self.x) - (self.counts + 1)*self.context_length, (self.batch_size, )) #done to prevent overshoot at any count
            batch = torch.stack([self.x[i : i + self.context_length] for i in inds])
            self.inds = inds
            self.counter = 0
            self.first_it = False
        
        elif self.training_xl and self.counter < self.counts:
            self.counter += 1
            batch = torch.stack([self.x[i + self.counter*self.context_length: i + (self.counter + 1)*self.context_length] for i in self.inds])
        
        elif (not val) and (not self.training_xl):
            inds = torch.randint(0, len(self.x) - self.context_length, (self.batch_size, ))
            batch = torch.stack([self.x[i : i + self.context_length] for i in inds])

        if val: # sample only one example from val set, overrides self.training_xl
            inds = torch.randint(0, len(self.x_val) - self.context_length, (1, ))
            batch = torch.stack([self.x_val[i : i + self.context_length] for i in inds])

        if self.training_xl and not val:
            return batch, self.counter
        
        return batch
        
def Train(model : Decoder or DecoderMemLM, data, optimizer):

    train_losses = {}
    val_losses = {}

    cur_min_loss = float('inf')
    if isinstance(model, Decoder):

        for epoch in range(4):
            for step in range(500000):

                batch = data.get_batch().to(DEVICE)
                loss, _, _ = model(batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses[epoch*500000 + step + 1] = loss.cpu().item()

                if (step + 1) % 1000 == 0:
                    print(f"Loss at epoch : {epoch + 1}, step : {step + 1} = {loss.cpu().item() :.4f}", flush=True)

                if (step + 1) % 1500 == 0:
                    model.eval()

                    eval_batch = data.get_batch(val=True).to(DEVICE)
                    loss, _, _ = model(eval_batch)
                    if loss < cur_min_loss:
                        cur_min_loss = loss
                        torch.save(model.state_dict(), f"chkpts/context-{CONTEXT_LENGTH-1}-lr-{LEARNING_RATE}.pth")
                    val_losses[epoch*500000 + step + 1] = loss.cpu().item()
                    print(f"Loss at epoch : {epoch + 1}, step : {step + 1} on validation set = {loss.cpu().item() :.4f}", flush=True)

                    model.train()

    else:

        past = None

        for epoch in range(3):
            for step in range(542000):
                batch, cur_cnt = data.get_batch()
                batch = batch.to(DEVICE)

                if cur_cnt == 0:
                    past = None
                    
                loss, _, past, _ = model(batch, past)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses[epoch*500000 + step + 1] = loss.cpu().item()

                if (step + 1) % 1000 == 0:
                    print(f"Loss at epoch : {epoch + 1}, step : {step + 1} = {loss.cpu().item() :.4f}", flush=True)

                if (step + 1) % 1500 == 0:
                    model.eval()

                    eval_batch = data.get_batch(val=True).to(DEVICE)
                    loss, _, _, _ = model(eval_batch)
                    if loss < cur_min_loss:
                        cur_min_loss = loss
                        torch.save(model.state_dict(), f"chkpts/context-{CONTEXT_LENGTH-1}-lr-{LEARNING_RATE}-xl-{COUNTS}.pth")
                    val_losses[epoch*500000 + step + 1] = loss.cpu().item()
                    print(f"Loss at epoch : {epoch + 1}, step : {step + 1} on validation set = {loss.cpu().item() :.4f}", flush=True)

                    model.train()
                
    return train_losses, val_losses

if __name__ == "__main__":

    model = DecoderMemLM(16000, CONTEXT_LENGTH, DEVICE)
    data = TweetSentiment(context_length=CONTEXT_LENGTH, training_xl=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses = Train(model, data, optimizer)
    if isinstance(model, DecoderMemLM):
        torch.save(model.state_dict(), f"state_dicts/context-{CONTEXT_LENGTH}-lr-{LEARNING_RATE}-xl-{COUNTS}.pth")
    else:
        torch.save(model.state_dict(), f"state_dicts/context-{CONTEXT_LENGTH}-lr-{LEARNING_RATE}.pth")
    
    plt.plot(train_losses.keys(), train_losses.values(), color='green', label="Train")
    plt.plot(val_losses.keys(), val_losses.values(), color='red', label="Validation")
    plt.legend()
    plt.show()
    if isinstance(model, DecoderMemLM):
        plt.savefig(f"plots/vocab-16k-context-{CONTEXT_LENGTH}-lr-{LEARNING_RATE}-xl-{COUNTS}.png")
    else:
        plt.savefig(f"plots/vocab-16k-context-{CONTEXT_LENGTH}-lr-{LEARNING_RATE}.png")