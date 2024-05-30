import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Adam
import torch.nn as nn
from DataEmbedding import DataEmbedding
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class LSTM(nn.Module):
    def __init__(self, input_size, num_of_classes, hidden_size, dropout, layers):
        super().__init__()
        self.num_of_classes = num_of_classes
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)


    def forward(self, sen_embeddings, sen_lens):
        packed = pack_padded_sequence(sen_embeddings, sen_lens, batch_first=True, enforce_sorted=False)
        lstm_out, (hidden, cell) = self.lstm(packed)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        x = self.fc(hidden)
        x = self.dropout(x)
        x = nn.functional.softmax(x)
        return x


def run(hidden_size, dropout, lr, train_loader, test_loader, optim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Training on GPU.")
    else:
        print("No GPU available, training on CPU.")
    best_accuracy = 0
    best_preds = []
    lstm_model = LSTM(100, 3, hidden_size, dropout, layers=2)
    lstm_model = lstm_model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    if optim == "Adam":
        optimizer = Adam(params=lstm_model.parameters(), lr=lr)
    if optim == "Adagrad":
        optimizer = torch.optim.Adagrad(params=lstm_model.parameters(), lr=lr)
    if optim == "RMSprop":
        optimizer = torch.optim.SGD(params=lstm_model.parameters(), lr=lr)
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    epochs = 100
    for epoch in range(epochs):
        all_preds_train = []
        train_loss = 0
        for index, (sens, labels, sen_lens) in enumerate(train_loader):
            sens = sens.to(device)
            labels = labels.to(device)
            o = lstm_model(sens, sen_lens)
            preds = o.detach().cpu().numpy()
            preds = [np.argmax(pred) for pred in preds]
            all_preds_train += preds
            loss = loss_fn(o, labels)
            train_loss += loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

        count = 0
        true_pred = 0
        for yi, pred in zip(y_train, all_preds_train):
            count += 1
            if yi == pred:
                true_pred += 1

        train_acc = true_pred / count
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss/len(train_loader.dataset))

        all_preds_test = []
        test_loss = 0
        for sens, labels, sen_lens in test_loader:
            sens = sens.to(device)
            labels = labels.to(device)
            oo = lstm_model(sens, sen_lens)
            loss = loss_fn(oo, labels)
            test_loss += loss.item()
            preds = oo.detach().cpu().numpy()
            preds = [np.argmax(pred) for pred in preds]
            all_preds_test += preds

        count = 0
        true_pred = 0
        for yi, pred in zip(y_test, all_preds_test):
            count += 1
            if yi == pred:
                true_pred += 1

        test_acc = true_pred / count
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss / len(test_loader.dataset))

        print(test_acc)
        if best_accuracy < test_acc:
            best_accuracy = test_acc
            best_preds = all_preds_test

    return best_accuracy, best_preds




seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


vec_size = 100
num_of_classes = 3
num_layers = 2


train_df = pd.read_csv('data/trainEmotions.csv')
test_df = pd.read_csv('data/testEmotions.csv')
print("Loading data")
embedding = DataEmbedding(vec_size)
train_loader, y_train = embedding.get_sen_embedding_from_path(train_df, True, "train")
test_loader, y_test = embedding.get_sen_embedding_from_path(test_df, True, "test")

lr = 0.001
#train_acc_list1, train_loss_list1, test_acc_list1, test_loss_list1 = run(128, 0.2, lr, train_loader, test_loader, "RMSprop")
best_accuracy, best_preds = run(128, 0.2, lr, train_loader, test_loader, "Adam")
#train_acc_list3, train_loss_list3, test_acc_list3, test_loss_list3 = run(128, 0.2, lr, train_loader, test_loader, "Adagrad")


#with open('lists.pkl', 'wb') as f:
#    pickle.dump(train_acc_list1, f)
#    pickle.dump(train_loss_list1, f)
#    pickle.dump(test_acc_list1, f)
#    pickle.dump(test_loss_list1, f)
#    pickle.dump(train_acc_list2, f)
#    pickle.dump(train_loss_list2, f)
#    pickle.dump(test_acc_list2, f)
#    pickle.dump(test_loss_list2, f)
#    pickle.dump(train_acc_list3, f)
#    pickle.dump(train_loss_list3, f)
#    pickle.dump(test_acc_list3, f)
#    pickle.dump(test_loss_list3, f)

#with open('lists.pkl', 'rb') as f:
#    train_acc_list1 = pickle.load(f)
#    train_loss_list1 = pickle.load(f)
#    test_acc_list1 = pickle.load(f)
#    test_loss_list1 = pickle.load(f)
#    train_acc_list2 = pickle.load(f)
#    train_loss_list2 = pickle.load(f)
#    test_acc_list2 = pickle.load(f)
#    test_loss_list2 = pickle.load(f)
#    train_acc_list3 = pickle.load(f)
#    train_loss_list3 = pickle.load(f)
#    test_acc_list3 = pickle.load(f)
#    test_loss_list3 = pickle.load(f)

#fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot each graph in the corresponding subplot
#axes[0, 0].plot(train_acc_list1, label='Train accuracy')
#axes[0, 0].plot(test_acc_list1, label='Test accuracy')
#axes[0, 0].set_xlabel('epoch')
#axes[0, 0].set_ylabel('Accuracy')
#axes[0, 0].set_title('Accuracy with RMSprop')
#axes[1, 0].plot(train_loss_list1, label='Train loss')
#axes[1, 0].plot(test_loss_list1, label='Test loss')
#axes[1, 0].set_title('Loss with RMSprop')
#axes[1, 0].set_xlabel('epoch')
#axes[1, 0].set_ylabel('Loss')
#axes[0, 1].plot(train_acc_list2, label='Train accuracy')
#axes[0, 1].plot(test_acc_list2, label='Train accuracy')
#axes[0, 1].set_title('Accuracy with Adam')
#axes[0, 1].set_xlabel('epoch')
#axes[0, 1].set_ylabel('Accuracy')
#axes[1, 1].plot(train_loss_list2, label='Train loss')
#axes[1, 1].plot(test_loss_list2, label='Test loss')
#axes[1, 1].set_title('Loss with Adam')
#axes[1, 1].set_xlabel('epoch')
#axes[1, 1].set_ylabel('Loss')
#axes[0, 2].plot(train_acc_list3, label='Train accuracy')
#axes[0, 2].plot(test_acc_list3, label='Test accuracy')
#axes[0, 2].set_title('Accuracy with Adagrad')
#axes[0, 2].set_xlabel('epoch')
#axes[0, 2].set_ylabel('Accuracy')
#axes[1, 2].plot(train_loss_list3, label='Train loss')
#axes[1, 2].plot(test_loss_list3, label='Test loss')
#axes[1, 2].set_title('Loss with Adagrad')
#axes[1, 2].set_xlabel('epoch')
#axes[1, 2].set_ylabel('Loss')
#for ax in axes.flat:
#    ax.legend()
#plt.savefig("optimizers.png")

print(best_accuracy)

conf_matrix = confusion_matrix(y_test, best_preds)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(np.arange(len(conf_matrix)), labels=['sadness', 'neutral', 'happiness'])
plt.yticks(np.arange(len(conf_matrix)), labels=['sadness', 'neutral', 'happiness'])
plt.tight_layout()

# Display the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)
plt.show()


