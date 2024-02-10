import torch.nn as nn

# Definisi kelas NeuralNet
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        # Layer input -> hidden layer
        self.l1 = nn.Linear(input_size, hidden_size)  # Linear layer dengan input_size neuron dan hidden_size neuron
        # Layer hidden layer -> hidden layer
        self.l2 = nn.Linear(hidden_size, hidden_size)  # Linear layer dengan hidden_size neuron dan hidden_size neuron
        # Layer hidden layer -> output layer
        self.l3 = nn.Linear(hidden_size, num_classes)  # Linear layer dengan hidden_size neuron dan num_classes neuron
        # Fungsi aktivasi ReLU: Menerapkan fungsi max(0, x) pada setiap elemen
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Forward pass melalui layer-layer dengan fungsi aktivasi ReLU
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        
        return out


    # input_size: Jumlah fitur atau dimensi dalam data input.
    #             Sesuaikan dengan jumlah fitur pada data input yang digunakan untuk melatih model.

    # hidden_size: Jumlah neuron atau unit dalam setiap hidden layer.
    #              Menentukan kapasitas dan kompleksitas model. 
    #              Nilai yang terlalu kecil atau besar dapat mempengaruhi kinerja model.

    # num_classes: Jumlah kelas atau kategori dalam tugas klasifikasi.
    #              Menentukan jumlah neuron pada output layer. 
    #              Sesuai dengan jumlah kelas dalam tugas klasifikasi.

    # self.l1, self.l2, self.l3: Layer-layer linear yang menghubungkan input ke hidden layer dan hidden layer ke output layer.
    #                            Merepresentasikan parameter yang dapat diubah (trainable) yang akan disesuaikan selama pelatihan
    #                            untuk memodelkan hubungan antara input dan output.

    # self.relu: Fungsi aktivasi ReLU yang diterapkan setelah setiap layer linear.
    #            Menambahkan non-linearitas ke model, memungkinkan pembelajaran pola yang lebih kompleks.
