import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from torchinfo import summary
import numpy as np
import sys
from tqdm import tqdm
import itertools
from TaPR_pkg import etapr


class DCNN_Dataset(Dataset):
    def __init__(self, x_data, y_label):
        self.x = x_data
        self.y = y_label

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx)
        x_idx = torch.from_numpy(self.x[idx])
        y_idx = torch.from_numpy(self.y[idx])
        item = {"given": x_idx, "answer": y_idx}
        return item

def conv2d_bn(in_channels, out_channels, k, s, p):
    return nn.Sequencial(nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p, stride=s),
                         nn.BatchNorm2d(out_channels), nn, ReLU())

class DCNN(nn.Module):
    def __init__(self):
        super(DCNN, self).__init__()

        self.relu = nn.ReLU()

        # For stem
        self.stem_1 = conv2d_bn(1, 32, k=3, s=1, p=1)
        self.stem_2 = conv2d_bn(32, 32, k=3, s=1, p=0)
        self.stem_3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.stem_4 = conv2d_bn(32, 64, k=1, s=1, p=0)
        self.stem_5 = conv2d_bn(64, 128, k=3, s=1, p=1)
        self.stem_6 = conv2d_bn(128, 128, k=3, s=1, p=1)

        # For Inception_resnet_A
        self.Inception_resnet_A_bn2 = conv2d_bn(128, 32, k=3, s=1, p=0)

        self.Inception_resnet_A_bn3_1 = conv2d_bn(128, 32, k=3, s=1, p=0)
        self.Inception_resnet_A_bn3_2 = conv2d_bn(32, 32, k=3, s=1, p=1)

        self.Inception_resnet_A_bn4_1 = conv2d_bn(128, 32, k=3, s=1, p=0)
        self.Inception_resnet_A_bn4_2 = conv2d_bn(32, 32, k=3, s=1, p=1)
        self.Inception_resnet_A_bn4_3 = conv2d_bn(128, 32, k=3, s=1, p=1)

        self.Inception_resnet_A_linear = nn.Conv2d(96, 128, k=1, s=1, p=0)


        # For Reduction-A
        self.Reduction_A_bn1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.Reduction_A_bn2 = conv2d_bn(128, 192, k=3, s=2, p=0)

        self.Reduction_A_bn3_1 = conv2d_bn(128, 96, k=1, s=1, p=0)
        self.Reduction_A_bn3_2 = conv2d_bn(96, 96, k=3, s=1, p=1)
        self.Reduction_A_bn3_3 = conv2d_bn(96, 128, k=3, s=2, p=0)


        # For Inception_resnet_B
        self.Inception_resnet_B_bn2 = conv2d_bn(448, 64, k=1, s=1, p=0)

        self.Inception_resnet_B_bn3_1 = conv2d_bn(448, 64, k=1, s=1, p=0)
        self.Inception_resnet_B_bn3_2 = conv2d_bn(64, 64, k=(1, 3), s=1, p=(0, 1))
        self.Inception_resnet_B_bn3_3 = conv2d_bn(64, 64, k=(3, 1), s=1, p=(1, 0))

        self.Inception_resnet_B_linear = nn.Conv2d(128, 448, k=1, s=1, p=0)


        # For Reduction-B
        self.Reduction_B_bn1 = nn.MaxPool2d(kernel_size=3, stride=3)

        self.Reduction_B_bn2_1 = conv2d_bn(448, 128, k=1, s=1, p=0)
        self.Reduction_B_bn2_2 = conv2d_bn(128, 192, k=3, s=3, p=0)

        self.Reduction_B_bn3_1 = conv2d_bn(448, 128, k=1, s=1, p=0)
        self.Reduction_B_bn3_2 = conv2d_bn(128, 192, k=3, s=3, p=0)

        self.Reduction_B_bn4_1 = conv2d_bn(448, 128, k=1, s=1, p=0)
        self.Reduction_B_bn4_2 = conv2d_bn(128, 128, k=3, s=1, p=1)
        self.Reduction_B_bn4_3 = conv2d_bn(128, 128, k=3, s=3, p=0)


        self.averpool = nn.AvgPool2d(kernel_size=2, stride=1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(896, 2)


    def forward(self, x):

        x = x.unsqueeze(3)
        x = np.transpose(x,(0,3,1,2))
        x = torch.tensor(x).float()

        #stem
        x = self.stem_1(x)
        x = self.stem_2(x)
        x = self.stem_3(x)
        x = self.stem_4(x)
        x = self.stem_5(x)
        x = self.stem_6(x)


        # For Inception_resnet_A

        Inception_resnet_A_bn1 = self.relu(x)

        Inception_resnet_A_bn2 = self.Inception_resnet_A_bn2(Inception_resnet_A_bn1)

        Inception_resnet_A_bn3_1 = self.Inception_resnet_A_bn3_1(Inception_resnet_A_bn1)
        Inception_resnet_A_bn3_2 = self.Inception_resnet_A_bn3_2(Inception_resnet_A_bn3_1)

        Inception_resnet_A_bn4_1 = self.Inception_resnet_A_bn4_1(Inception_resnet_A_bn1)
        Inception_resnet_A_bn4_2 = self.Inception_resnet_A_bn4_2(Inception_resnet_A_bn4_1)
        Inception_resnet_A_bn4_3 = self.Inception_resnet_A_bn4_3(Inception_resnet_A_bn4_2)

        Inception_resnet_A_concat = torch.cat((Inception_resnet_A_bn2,Inception_resnet_A_bn3_2,Inception_resnet_A_bn4_3 ),1)
        Inception_resnet_A_linear = self.Inception_resnet_A_linear(Inception_resnet_A_concat)

        Inception_resnet_A_sum = Inception_resnet_A_bn1+Inception_resnet_A_linear
        x = self.relu(Inception_resnet_A_sum)

        # For Reduction_A

        Reduction_A_bn1 = self.Reduction_A_bn1(x)

        Reduction_A_bn2 = self.Reduction_A_bn2(x)

        Reduction_A_bn3_1 = self.Reduction_A_bn3_1(x)
        Reduction_A_bn3_2 = self.Reduction_A_bn3_2(Reduction_A_bn3_1)
        Reduction_A_bn3_3 = self.Reduction_A_bn3_3(Reduction_A_bn3_2)

        x = torch.cat((Reduction_A_bn1,Reduction_A_bn2,Reduction_A_bn3_3),1)


        # For Inception_resnet_B

        Inception_resnet_B_bn1 = self.relu(x)

        Inception_resnet_B_bn2 = self.Inception_resnet_B_bn2(Inception_resnet_B_bn1)

        Inception_resnet_B_bn3_1 = self.Inception_resnet_B_bn3_1(Inception_resnet_B_bn1)
        Inception_resnet_B_bn3_2 = self.Inception_resnet_B_bn3_2(Inception_resnet_B_bn3_1)
        Inception_resnet_B_bn3_3 = self.Inception_resnet_B_bn3_3(Inception_resnet_B_bn3_2)

        Inception_resnet_B_concat = torch.cat((Inception_resnet_B_bn2,Inception_resnet_B_bn3_3),1)
        Inception_resnet_B_linear = self.Inception_resnet_B_linear(Inception_resnet_B_concat)

        Inception_resnet_B_sum = Inception_resnet_B_bn1+Inception_resnet_B_linear

        x= self.relu(Inception_resnet_B_sum)


        # For Reduction_B

        Reduction_B_bn1 = self.Reduction_B_bn1(x)

        Reduction_B_bn2_1 = self.Reduction_B_bn2_1(x)
        Reduction_B_bn2_2 = self.Reduction_B_bn2_2(Reduction_B_bn2_1)

        Reduction_B_bn3_1 = self.Reduction_B_bn3_1(x)
        Reduction_B_bn3_2 = self.Reduction_B_bn3_2(Reduction_B_bn3_1)

        Reduction_B_bn4_1 = self.Reduction_B_bn4_1(x)
        Reduction_B_bn4_2 = self.Reduction_B_bn4_2(Reduction_B_bn4_1)
        Reduction_B_bn4_3 = self.Reduction_B_bn4_3(Reduction_B_bn4_2)

        x = torch.cat(( Reduction_B_bn1,Reduction_B_bn2_2,Reduction_B_bn3_2,Reduction_B_bn4_3),1)


        #last

        x = self.averpool(x)
        x = self.dropout(x)
        x = x.squeeze()
        x = self.fc(x)
        Y = nn.functional.softmax(x,dim=1)

        return Y

def train(dataset,model,batch_size,n_epochs):
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=0.0001)
    loss_fn = torch.nn.BCELoss()
    epochs = tqdm(range(n_epochs))
    best = {"loss":sys.float_info.max}
    loss_histoty=[]
    for e in epochs:
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            given = batch["given"].cuda()
            guess = model(given)
            answer = batch["answer"].cuda()
            loss = loss_fn(guess,answer)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        loss_histoty.append(epoch_loss)
        epochs.set_postfix_str(f"loss:{epoch_loss:.6f}")
        if epoch_loss < best["loss"]:
            best["state"] = model.state_dict()
            best["loss"] = epoch_loss
            best["epoch"] = e+1
    return best, loss_histoty

def inference(dataset, model, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    guess_prob_list = []
    guess_arg_list = []
    answer_prob_list = []
    answer_arg_list = []
    with torch.no_grad():
        for batch in dataloader:
            given = batch["given"].cuda()
            guess = model(given)
            answer = batch["answer"].cuda()
            guess_prob, guess_arg = torch.max(guess,1)
            answer_prob,answer_arg = torch.max(answer,1)
            guess_prob_list.append(guess_prob.tolist())
            guess_arg_list.append(guess_arg.tolist())
            answer_prob_list.append(answer_prob.tolist())
            answer_arg_list.append(answer_arg.tolist())
    guess_prob_list = list(itertools.chain(*guess_prob_list))
    guess_arg_list = list(itertools.chain(*guess_arg_list))
    answer_prob_list = list(itertools.chain(*answer_prob_list))
    answer_arg_list = list(itertools.chain(*answer_arg_list))

    return guess_prob_list,guess_arg_list,answer_prob_list,answer_arg_list

batch=1000

train_A_x=np.load("path")
train_B_x=np.load("path")

train_A_y=np.load("path")
train_B_y=np.load("path")

test_x=np.load("path")
test_y=np.load("path")

train_x=np.concatenate([train_A_x,train_B_x],axis=0)
train_y=np.concatenate([train_A_y,train_B_y],axis=0)

DCNN_DATASET = DCNN_Dataset(x_data=train_x,y_label=train_y)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=DCNN.to(device)
model.train()

BEST_MODEL, LOSS_HISTORY = train(DCNN_DATASET, model, batch, 100)

with open("DCNN_train",'wb') as f:
    torch.save({"state":BEST_MODEL["state"], "best_epoch":BEST_MODEL["epoch"],"loss_history":LOSS_HISTORY},f)



DCNN_DATASET_TEST = DCNN_Dataset(x_data=test_x,y_label=test_y)

with open("DCNN_train",'rb') as f:
    SAVED_MODEL = torch.load(f)

model.load_state_dict(SAVED_MODEL["state"])
model.eval()

guess_prob_list,guess_arg_list,answer_prob_list,answer_arg_list = inference(DCNN_DATASET_TEST,model,batch_size=batch)

TaPR = etapr.evaluate(anomalies=answer_arg_list,predictions=guess_arg_list)
print(f"F1: {TaPR['f1']:.4f}")




    
    
    
    
