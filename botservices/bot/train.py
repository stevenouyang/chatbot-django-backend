import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Memuat data dari file JSON
with open('train-intents.json', 'r') as f:
    intents = json.load(f)

# Inisialisasi list untuk menyimpan kata-kata, tag, dan data latihan
all_words = []
tags = []
xy = []

# Memproses setiap intent dan polanya
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Menghapus tanda baca dan mengubah kata ke bentuk dasar
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Menampilkan informasi dataset
print(len(xy), "polanya")
print(len(tags), "tag:", tags)
print(len(all_words), "kata yang diubah ke bentuk dasar:", all_words)

# Menyiapkan data latihan (X_train dan y_train)
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyperparameter
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

# Mendefinisikan dataset khusus untuk pelatihan
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Membuat DataLoader untuk dataset pelatihan
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Menetapkan perangkat untuk pelatihan (menggunakan GPU jika tersedia)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Menginisialisasi model jaringan saraf
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Mendefinisikan fungsi loss dan optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Melatih model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward dan optimasi
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Menampilkan loss setiap 100 epoch
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Menampilkan loss akhir
print(f'Loss akhir: {loss.item():.4f}')

# Menyimpan model dan data terkait
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Pelatihan selesai. Model disimpan di {FILE}')
