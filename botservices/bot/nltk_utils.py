import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

# Inisialisasi PorterStemmer
stemmer = PorterStemmer()

# Tokenisasi kalimat menjadi kata-kata
def tokenize(kalimat):
    return nltk.word_tokenize(kalimat)

# Menyederhanakan kata menggunakan PorterStemmer dan mengonversi menjadi huruf kecil
def stem(kata):
    return stemmer.stem(kata.lower())

# Membuat representasi "bag of words" untuk kalimat yang sudah di-tokenisasi
def bag_of_words(kalimat_token, kata_kunci):
    # Menyederhanakan setiap kata dalam kalimat yang sudah di-tokenisasi
    kata_kunci_kalimat = [stem(kata) for kata in kalimat_token]
    
    # Inisialisasi "bag" dengan nilai nol
    bag = np.zeros(len(kata_kunci), dtype=np.float32)
    
    # Setel indeks yang sesuai menjadi 1 jika kata kunci tersebut ada dalam kalimat
    for idx, kata_kunci in enumerate(kata_kunci):
        if kata_kunci in kata_kunci_kalimat: 
            bag[idx] = 1

    return bag

# Contoh input kalimat
kalimat_contoh = "Ini adalah contoh kalimat untuk di-tokenisasi."

# Tokenisasi kalimat
kalimat_token = tokenize(kalimat_contoh)
# Output tokenisasi: ['Ini', 'adalah', 'contoh', 'kalimat', 'untuk', 'di-tokenisasi', '.']

# Contoh input kata kunci
kata_kunci_contoh = ["Ini", "adalah", "contoh", "kalimat", "sederhana"]
# Input kata kunci biasanya didapat dari kumpulan kata unik dalam data latih.

# Representasi "Bag of Words" untuk kalimat input
bag_of_words_contoh = bag_of_words(kalimat_token, kata_kunci_contoh)
# Output bag of words: [1.0, 1.0, 1.0, 1.0, 0.0]
# Setiap nilai dalam array menunjukkan keberadaan (1) atau ketiadaan (0) kata kunci pada kalimat input.
# Urutan nilai sesuai dengan urutan kata kunci pada list input.
