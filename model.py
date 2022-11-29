import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        if torch.cuda.is_available():
            print("GPU")  
            self.dev = "cuda:0" 
        else:  
            print("CPU")
            self.dev = "cpu" 
            self.linear1 = nn.Linear(input_size, hidden_size).to(self.dev)
            self.linear15 = nn.Linear(hidden_size,hidden_size).to(self.dev)
            self.linear25 = nn.Linear(hidden_size,hidden_size).to(self.dev)
            self.linear35 = nn.Linear(hidden_size,hidden_size).to(self.dev)
            self.linear45 = nn.Linear(hidden_size,hidden_size).to(self.dev)
            self.linear55 = nn.Linear(hidden_size,hidden_size).to(self.dev)
            self.linear65 = nn.Linear(hidden_size,hidden_size).to(self.dev)
            self.linear75 = nn.Linear(hidden_size,hidden_size).to(self.dev)
            self.linear85 = nn.Linear(hidden_size,hidden_size).to(self.dev)
            self.linear95 = nn.Linear(hidden_size,hidden_size).to(self.dev)
            self.linear05 = nn.Linear(hidden_size,hidden_size).to(self.dev)
            self.linear2 = nn.Linear(hidden_size, output_size).to(self.dev)

    def forward(self, x):
        x = F.relu(self.linear1(x)).to(self.dev)
        x = self.linear15(x)
        x = self.linear25(x)
        x = self.linear35(x)
        x = self.linear45(x)
        x = self.linear55(x)
        x = self.linear65(x)
        x = self.linear75(x)
        x = self.linear85(x)
        x = self.linear95(x)
        x = self.linear05(x)
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        if torch.cuda.is_available():  
            self.dev = "cuda:0" 
        else:  
            self.dev = "cpu"  
        self.lr = lr
        self.gamma = gamma
        self.model = model.to(self.dev)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss().to(self.dev)

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).to(self.dev)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.dev)
        action = torch.tensor(action, dtype=torch.long).to(self.dev)
        reward = torch.tensor(reward, dtype=torch.float).to(self.dev)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0).to(self.dev)
            next_state = torch.unsqueeze(next_state, 0).to(self.dev)
            action = torch.unsqueeze(action, 0).to(self.dev)
            reward = torch.unsqueeze(reward, 0).to(self.dev)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])).to(self.dev)

            target[idx][torch.argmax(action[idx]).item()] = Q_new.to(self.dev)
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()



