'''
Finetuner.
'''

from transformer_base import Decoder
from transformer_xl import DecoderMemLM
from data.sts_data_cleaner import STSDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from tqdm import tqdm
import sentencepiece as spm
import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 6e-5
AUX = 0.5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOKENIZER = spm.SentencePieceProcessor("tokenizer/tweets-16.model")

class MyCollator(object):

    def __init__(self, model : str):
        self.model = model

    def __call__(self, batch):
        '''
        Collate function for the dataloader.
        Produces both sentences for the model with the extract token as the first token.

        #NOTE : upon further inspection of the dataset, I saw that increasing the context
        length from 64 to 128 covers almost all sentences completely (except 4). So for 
        transformer-xl I will use 128 as the context length, i.e. do two forward passes 
        for each sentence.
        '''

        extract = TOKENIZER.PieceToId('<e>')

        if self.model == 'base':
            
            sentences = []
            sentences_T = []
            targets = []
            for sample in batch:

                sentence = TOKENIZER.EncodeAsIds(sample[0], add_bos=True, add_eos=True)
                sentence_T = TOKENIZER.EncodeAsIds(sample[1], add_bos=True, add_eos=True)

                if len(sentence) < 64:
                    sentence = sentence[:-1] + [TOKENIZER.pad_id()]*(63 - len(sentence)) + [extract] + [sentence[-1]]
                    sentence_T = sentence_T[:-1] + [TOKENIZER.pad_id()]*(63 - len(sentence_T)) + [extract] + [sentence_T[-1]]
                else:
                    sentence = sentence[:63] + [extract]
                    sentence_T = sentence_T[:63] + [extract]

                assert len(sentence) == 64
                assert len(sentence_T) == 64

                sentences.append(torch.tensor(sentence).reshape(1, -1))
                sentences_T.append(torch.tensor(sentence_T).reshape(1, -1))
                targets.append(torch.tensor(sample[2], dtype=torch.float32).reshape(1,))

            batch_sentence = torch.cat(sentences, dim=0)
            batch_sentence_T = torch.cat(sentences_T, dim=0)
            batch_targets = torch.cat(targets, dim=0)

        elif self.model == 'xl':
            '''
            basically, exactly same with context length of 128, other logic is in the rest of the code.
            '''
            sentences = []
            sentences_T = []
            targets = []
            for sample in batch:

                sentence = TOKENIZER.EncodeAsIds(sample[0], add_bos=True, add_eos=True)
                sentence_T = TOKENIZER.EncodeAsIds(sample[1], add_bos=True, add_eos=True)

                if len(sentence) < 128:
                    sentence = sentence[:-1] + [TOKENIZER.pad_id()]*(62 - len(sentence)) + [extract] 
                    sentence = sentence + [TOKENIZER.pad_id()]*(126 - len(sentence)) + [extract] + [sentence[-1]]
                    sentence_T = sentence_T[:-1] + [TOKENIZER.pad_id()]*(62 - len(sentence_T)) + [extract]
                    sentence_T = sentence_T + [TOKENIZER.pad_id()]*(126 - len(sentence_T)) + [extract] + [sentence_T[-1]]
                else:
                    sentence = sentence[:-1] + [extract] + [sentence[-1]]
                    sentence_T = sentence_T[:-1] + [extract] + [sentence_T[-1]]

                assert len(sentence) == 128, f"{len(sentence)}"
                assert len(sentence_T) == 128
                assert sentence[-1] == extract
                assert sentence_T[-1] == extract

                sentences.append(torch.tensor(sentence).reshape(1, -1))
                sentences_T.append(torch.tensor(sentence_T).reshape(1, -1))
                targets.append(torch.tensor(sample[2], dtype=torch.float32).reshape(1,))

            batch_sentence = torch.cat(sentences, dim=0)
            batch_sentence_T = torch.cat(sentences_T, dim=0)
            batch_targets = torch.cat(targets, dim=0)

        return (batch_sentence, batch_sentence_T, batch_targets)

class Finetune:

    def __init__(self, 
                 model : Decoder or DecoderMemLM, 
                 task : str, 
                 eval_every : int):
        
        '''
        The paper uses a single weight i.e. nn.Parameter, but my model isn't expressive enough,
        so I use a small MLP, which is -
        1. nn.Linear(self.model.hidden_dim, 2*self.model.hidden_dim)
        2. nn.ReLU(inplace=True)
        3. nn.Linear(2*self.model.hidden_dim, 1)
        '''

        self.model = model
        self.task = task
        self.eval_every = eval_every
        #Final Prediction network
        self.network = nn.Linear(self.model.hidden_dim, 1)
        # self.network = nn.Sequential(
        #     nn.Linear(self.model.hidden_dim, 2*self.model.hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(2*self.model.hidden_dim, 1)
        #     ).to(DEVICE)

    def finetune(self, 
                 trainDataloader : DataLoader, 
                 valDataloader : DataLoader,
                 testDataloader : DataLoader,
                 epochs : int = 3):
        '''
        Finetunes the model on the given task. (only similarity in my case)
        '''

        #Set up parameters
        criterion = nn.MSELoss()
        optimizer = optim.Adam(list(self.model.parameters()) + list(self.network.parameters()), lr = LEARNING_RATE) #[self.weight]
        run_num = torch.randint(0, 100, (1,)).item()
        print(f"RUN NUMBER : {run_num}", flush=True)

        #For saving
        best_model = self.model
        best_weight = self.network
        cur_max_coeff = -float('inf')

        #For plotting
        train_losses = []
        val_pcs = []

        if isinstance(self.model, DecoderMemLM):

            for _ in range(epochs):

                print("!Starting new epoch!", flush=True)
            
                for i, (sentence, sentence_T, target) in tqdm(enumerate(trainDataloader)):

                    #First pass the first 64 tokens through the model AND then the next 64 tokens
                    #with the first 64 frozen.
                    #My comments are more verbose for the other model!
                    loss1, _, new_past1, hid1 = self.model(sentence[:, :64].to(DEVICE), past=None, finetune=True)
                    #loss2, _, _, hid2 = self.model(sentence[:, 64:].to(DEVICE), past=new_past1, finetune=True, past_text=sentence[:, :64].to(DEVICE))

                    loss3, _, new_past2, hid3 = self.model(sentence_T[:, :64].to(DEVICE), past=None, finetune=True)
                    #loss4, _, _, hid4 = self.model(sentence_T[:, 64:].to(DEVICE), past=new_past2, finetune=True, past_text=sentence_T[:, :64].to(DEVICE))

                    logit = (hid1 + hid3)/2 # hid2 + hid4 
                    pred = self.network(logit).squeeze(1)

                    loss = criterion(pred, target.to(DEVICE)) + AUX*(loss1 + loss3) #+ loss2 + loss4

                    train_losses.append(loss.cpu().item())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if i % self.eval_every == 0:

                        with torch.no_grad():
                            self.model.eval()
                            self.network.eval()
                            s, s_t, t = next(iter(valDataloader))

                            _, _, new_past1, hid1 = self.model(s[:, :64].to(DEVICE), past=None, finetune=True)
                            #_, _, _, hid2 = self.model(s[:, 64:].to(DEVICE), past=new_past1, finetune=True, past_text=s[:, :64].to(DEVICE))

                            _, _, new_past2, hid3 = self.model(s_t[:, :64].to(DEVICE), past=None, finetune=True)
                            #_, _, _, hid4 = self.model(s_t[:, 64:].to(DEVICE), past=new_past2, finetune=True, past_text=s_t[:, :64].to(DEVICE))

                            logit = (hid1 + hid3)/2 #+ hid2 + hid4
                            pred = self.network(logit).squeeze(1)

                            pred = np.array(pred.tolist())
                            targets = np.array(t.tolist())
                            cur_pc = np.corrcoef(pred, targets)[0, 1]
                            print(f"Pearson's Coefficient at step = {i} is : {cur_pc :.4f}", flush=True)

                            if cur_pc > cur_max_coeff:
                                cur_max_coeff = cur_pc
                                best_model = self.model
                                best_weight = self.network
                                torch.save(self.network.state_dict(), f'chkpts/sentence-similarity/run-{run_num}-weight-aux-{AUX}-lr-{LEARNING_RATE}-xl.pth')
                                torch.save(self.model.state_dict(), f'chkpts/sentence-similarity/run-{run_num}-model-aux-{AUX}-lr-{LEARNING_RATE}-xl.pth')

                            val_pcs.append(cur_pc)
                            self.model.train()
                            self.network.train()

            print("!Fine Tuning Done!", flush=True) #, flush=True
            torch.save(self.model.state_dict(), f'state_dicts/finetuned-sentence-similarity-run-{run_num}-aux-{AUX}-lr-{LEARNING_RATE}-xl.pth')
            torch.save(self.network.state_dict(), f'state_dicts/weight-finetuned-sentence-similarity-run-{run_num}-aux-{AUX}-lr-{LEARNING_RATE}-xl.pth')

            print("Evaluating on test dataset!", flush=True)
            self.model = best_model
            self.network = best_weight

            self.model.eval()
            self.network.eval()

            preds = []
            targets = []

            with torch.no_grad():

                for i, (sentence, sentence_T, target) in tqdm(enumerate(testDataloader)):

                    _, _, new_past1, hid1 = self.model(sentence[:, :64].to(DEVICE), past=None, finetune=True)
                    #_, _, _, hid2 = self.model(sentence[:, 64:].to(DEVICE), past=new_past1, finetune=True)

                    _, _, new_past2, hid3 = self.model(sentence_T[:, :64].to(DEVICE), past=None, finetune=True)
                    #_, _, _, hid4 = self.model(sentence_T[:, 64:].to(DEVICE), past=new_past2, finetune=True)

                    logit = (hid1 + hid3)/2 #hid2 + hid4
                    pred = self.network(logit).squeeze(1)

                    for i in range(len(pred)):
                        preds.append(pred.tolist()[i])
                        targets.append(target.tolist()[i])

            preds = np.array(preds)
            targets = np.array(targets)

            final_pc = np.corrcoef(preds, targets)[0, 1]
            print(f"FINAL PEARSON'S COEFFICIENT ON TEST SET : {final_pc : .4f}")
        
        elif isinstance(self.model, Decoder):

            for _ in range(epochs):

                print("!Starting new epoch!", flush=True)

                for i, (sentence, sentence_T, target) in tqdm(enumerate(trainDataloader)):

                    #According to the paper, the order of the sentences should not matter
                    #So they used this training strategy, which I will also use; however my
                    #model overfits to the size of the input i.e. it EXPECTS that the input
                    #will be 2*normal input.

                    loss1, _, hid1 = self.model(sentence.to(DEVICE), finetune=True)
                    loss2, _, hid2 = self.model(sentence_T.to(DEVICE), finetune=True)
                    logit = (hid1 + hid2)/2

                    pred = self.network(logit)
                    pred = pred.to(DEVICE)

                    #Auxiliary loss actually helps!
                    loss = criterion(pred.squeeze(1), target.to(DEVICE)) + AUX*(loss1 + loss2)

                    #For plotting
                    train_losses.append(loss.cpu().item())

                    #GD Step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    #Eval code
                    if i % self.eval_every == 0:
                        with torch.no_grad():
                            self.model.eval()
                            s, s_t, t = next(iter(valDataloader))
                            _, _, hid1 = self.model(s.to(DEVICE), finetune=True)
                            _, _, hid2 = self.model(s_t.to(DEVICE), finetune=True)

                            logit = (hid1 + hid2)/2

                            pred = self.network(logit)

                            pred = pred.squeeze(1).cpu() #because shape would otherwise be (bs, 1)

                            pred = np.array(pred.tolist())
                            targets = np.array(t.tolist())

                            cur_pc = np.corrcoef(targets, pred)[0, 1]
                            print(f"Pearson's Coefficient at step = {i} is : {cur_pc :.4f}", flush=True)
                            val_pcs.append(cur_pc)

                            #Saving
                            if cur_pc > cur_max_coeff:
                                best_model = self.model
                                best_weight = self.network #weight
                                cur_max_coeff = cur_pc
                                torch.save(self.network.state_dict(), f'chkpts/sentence-similarity/run-{run_num}-weight-aux-{AUX}-lr-{LEARNING_RATE}.pth')
                                torch.save(self.model.state_dict(), f'chkpts/sentence-similarity/run-{run_num}-model-aux-{AUX}-lr-{LEARNING_RATE}.pth')

            print("!Fine Tuning Done!", flush=True)
            torch.save(self.model.state_dict(), f'state_dicts/finetuned-sentence-similarity-run-{run_num}-aux-{AUX}-lr-{LEARNING_RATE}.pth')
            torch.save(self.network.state_dict(), f'state_dicts/weight-finetuned-sentence-similarity-run-{run_num}-aux-{AUX}-lr-{LEARNING_RATE}.pth')

            print("Evaluating on test dataset!", flush=True)
            self.model = best_model
            self.network = best_weight

            self.model.eval()
            self.network.eval() #requires_grad_(False)

            preds = []
            targets = []

            with torch.no_grad():

                for i, (sentence, sentence_T, target) in tqdm(enumerate(testDataloader)):

                    _, _, hid1 = self.model(sentence.to(DEVICE), finetune=True)
                    _, _, hid2 = self.model(sentence_T.to(DEVICE), finetune=True)
                    logit = (hid1 + hid2)/2

                    pred = self.network(logit)

                    pred = pred.squeeze(1)

                    for i in range(len(pred)): #wasn't working as a generator object :/
                        preds.append(pred.cpu().tolist()[i])
                        targets.append(target.tolist()[i])

            #Final eval code
            preds = np.array(preds)
            targets = np.array(targets)
            final_pc = np.corrcoef(preds, targets)[0, 1]
            print(f"FINAL PEARSON'S COEFFICIENT ON TEST SET : {final_pc : .4f}", flush=True)

        return best_model, best_weight, train_losses, val_pcs, final_pc
        
if __name__ == "__main__":

    model = DecoderMemLM(16000, 65, DEVICE)
    model.load_state_dict(torch.load('state_dicts/context-65-lr-0.0001-xl-1.pth', map_location='cpu'))
    print('Model loaded!', flush=True)
    if isinstance(model, Decoder):
        collate_fn = MyCollator('base')
    else:
        collate_fn = MyCollator('xl')

    trainDataLoader = DataLoader(STSDataset(), batch_size=32, collate_fn=collate_fn)
    valDataLoader = DataLoader(STSDataset('dev'), batch_size=32, collate_fn=collate_fn)
    testDataLoader = DataLoader(STSDataset('test'), batch_size=32, collate_fn=collate_fn)

    finetuner = Finetune(model, 'similarity', 1)
    best_model, best_weight, train_losses, val_pcs, final_pc = finetuner.finetune(trainDataloader=trainDataLoader, valDataloader=valDataLoader, testDataloader=testDataLoader, epochs=3)

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(train_losses)
    ax1.set_title("Train Losses vs steps")

    ax2.plot(val_pcs)
    ax2.plot(final_pc, 'ro')
    ax2.set_title("Validation PCs vs steps")

    plt.savefig(f'plots/sentence-similarity-xl-lr-{LEARNING_RATE}.png')
    plt.close('all')