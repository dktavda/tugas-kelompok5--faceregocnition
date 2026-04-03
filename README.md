# tugas-kelompok5--faceregocnition

**Mata Kuliah:** Praktikum Machine Learning (Computer Vision)  
**Dosen Pengampu:** Herfandi, Ph.D. — Informatika — Universitas Teknologi Sumbawa (UTS)

---

## Profil Kelompok

* **[[Sherly Novia Indriani]** - [231001057]
* **[Dinda Oktavia Pratiwi]** - [231001026]
* **[Ahsanul Ikram]** - [231001077]
* **[Wanda Setya Budi]** - [231001008] 

---

## Deskripsi Proyek

Repositori ini merupakan dokumentasi teknis hasil praktik kelompok mengenai **Face Recognition menggunakan OpenCV** pada Python. Praktikum ini membahas cara membangun sistem pengenalan wajah secara lengkap mulai dari persiapan dataset, deteksi wajah, pelatihan model, evaluasi, hingga implementasi real-time menggunakan webcam.

Materi praktik merujuk pada implementasi tiga teknik face recognition yang tersedia di OpenCV, yaitu **Eigenface**, **Fisherface**, dan **Local Binary Pattern Histograms (LBPH)**, dengan fokus utama pada algoritma LBPH yang dipilih karena ketangguhannya terhadap variasi pencahayaan dan ekspresi wajah.

---

## Lingkungan Pengembangan (Environment)

Seluruh skrip dalam repositori ini disusun dan diuji menggunakan spesifikasi berikut:

* **Kode Editor:** Visual Studio Code (VSCode)
* **Interpreter:** Python 3.8.6
* **Environment Manager:** Anaconda (`opencv_env`)
* **Metode Eksekusi:** Cell-by-cell pada Jupyter Notebook / langsung di VSCode

---

## Library yang Digunakan

| Library | Versi | Fungsi |
|---------|-------|--------|
| `opencv-contrib-python` | 4.4.0.46 | Pengolahan gambar, deteksi & pengenalan wajah |
| `numpy` | 1.19.2 | Operasi array dan matriks |
| `matplotlib` | — | Visualisasi gambar dan grafik |
| `scikit-learn` | 0.23.2 | Evaluasi model (Confusion Matrix, Classification Report) |
| `itertools` | bawaan Python | Iterasi kombinasi untuk plot confusion matrix |
| `os` | bawaan Python | Operasi file dan folder |

Instalasi library:
```bash
pip install opencv-contrib-python
pip install scikit-learn
```

---

## Struktur Folder

```
pertemuan_5/
├── dataset/
│   ├── Colin_Powell/
│   ├── Donald_Rumsfeld/
│   ├── George_W_Bush/
│   ├── Gerhard_Schroeder/
│   ├── Tony_Blair/
│   └── Yunus/
├── test/
│   └── (5 gambar per orang untuk testing)
├── haarcascades/
│   └── haarcascade_frontalface_default.xml
├── my_face/
│   └── (foto wajah yang diambil dari webcam)
├── lbph_model.yml
└── pertemuan_5.ipynb
```

---

## Cakupan Praktikum

### 👤 Bagian 1 — Instalasi & Persiapan Data
Tahapan yang dilakukan:
1. **Instalasi library** yang dibutuhkan via `pip install`
2. **Import library** `os`, `cv2`, `numpy`, dan `matplotlib`
3. **Load dataset** dari folder secara otomatis menggunakan `os.listdir`, dibatasi 70 gambar per kelas
4. **Fungsi `show_dataset`** untuk menampilkan 5 sampel gambar per kelas
5. **Label Encoding** — mengubah nama string (misal `Colin_Powell`) menjadi angka integer menggunakan `np.unique` dan `np.where`, karena model ML hanya menerima input angka

### 👤 Bagian 2 — Deteksi & Preprocessing Wajah
Tahapan yang dilakukan:
1. **Load Haar Cascade** dari file `haarcascade_frontalface_default.xml` untuk mendeteksi wajah
2. **Fungsi `detect_face`** — mengkonversi gambar ke grayscale, mendeteksi wajah dengan `detectMultiScale`, lalu memotong (*crop*) dan mengubah ukuran (*resize*) area wajah menjadi 100×100 piksel
3. **Proses seluruh dataset** — setiap gambar diproses oleh `detect_face`; jika wajah tidak ditemukan, gambar dan labelnya dihapus agar data tetap sinkron
4. **Verifikasi hasil** dengan menampilkan gambar wajah yang sudah di-crop menggunakan `show_dataset`

### 👤 Bagian 3 — Training, Simpan Model & Testing
Tahapan yang dilakukan:
1. **Pembuatan model LBPH** menggunakan `cv2.face.LBPHFaceRecognizer_create()` (tersedia juga Eigenface dan Fisherface)
2. **Training model** dengan `model.train(croped_images, name_vec)` menggunakan gambar wajah dan label angkanya
3. **Simpan model** ke file `lbph_model.yml` dan memuat kembali dengan `model.read`
4. **Testing 1 gambar** — membaca gambar, deteksi wajah, prediksi dengan `model.predict`, dan menampilkan nama serta nilai confidence
5. **Testing semua gambar** di folder `test/` — hasil disimpan ke `actual_names`, `predicted_names`, dan `confidences` untuk evaluasi

### 👤 Bagian 4 — Evaluasi Model & Realtime Webcam
Tahapan yang dilakukan:
1. **Confusion Matrix** — memvisualisasikan hasil prediksi vs label asli dalam bentuk tabel berwarna menggunakan `sklearn.metrics.confusion_matrix`
2. **Classification Report** — menampilkan nilai Precision, Recall, F1-Score, dan Support per kelas menggunakan `sklearn.metrics.classification_report`
3. **Fungsi `draw_ped`** — menggambar kotak deteksi wajah dan label nama di atas frame video secara real-time
4. **Realtime Webcam** — membuka kamera, mendeteksi wajah di setiap frame, memprediksi identitas, dan menampilkan hasilnya secara langsung

---

## Teknik Face Recognition yang Dipelajari

### 1. Eigenface
Menggunakan **Principal Component Analysis (PCA)** untuk mencari representasi wajah berdimensi rendah. PCA mereduksi dimensi data gambar yang sangat tinggi (misalnya gambar 100×100 = 10.000 dimensi) menjadi komponen-komponen utama yang paling berpengaruh.

### 2. Fisherface
Menggunakan **Linear Discriminant Analysis (LDA)** sebagai pengembangan dari Eigenface. Jika PCA hanya memaksimalkan variansi total, LDA juga mempertimbangkan pemisahan antar kelas sehingga hasil klasifikasi lebih baik.

### 3. LBPH (Local Binary Pattern Histograms) ✅ *Digunakan*
Menganalisis **pola tekstur lokal** pada gambar wajah. Setiap piksel dibandingkan dengan 8 tetangganya untuk menghasilkan kode biner, lalu dirangkum dalam histogram. LBPH dipilih karena lebih tahan terhadap perubahan pencahayaan dan tidak mengharuskan semua gambar berukuran sama secara mutlak.

---

## Hasil Evaluasi Model

Model LBPH diuji pada **30 gambar** (5 gambar per kelas) dan menghasilkan:

```
== Classification Report for Test Dataset ==

                   precision    recall  f1-score   support

     Colin_Powell       1.00      1.00      1.00         5
  Donald_Rumsfeld       0.83      1.00      0.91         5
    George_W_Bush       1.00      1.00      1.00         5
Gerhard_Schroeder       1.00      1.00      1.00         5
       Tony_Blair       1.00      0.80      0.89         5
            Yunus       1.00      1.00      1.00         5

         accuracy                           0.97        30
        macro avg       0.97      0.97      0.97        30
     weighted avg       0.97      0.97      0.97        30
```

**Akurasi keseluruhan: 97%** — Satu kesalahan terjadi pada gambar `Tony_Blair_0142.jpg` yang diprediksi sebagai `Donald_Rumsfeld`.

---

## Analisis & Kesimpulan Teknis

1. **LBPH Unggul untuk Kondisi Variatif:** Dari ketiga algoritma yang tersedia, LBPH terbukti paling praktis untuk dataset wajah dengan variasi pencahayaan dan ekspresi. Nilai confidence mendekati 0.0 pada banyak gambar menunjukkan model sangat yakin dengan prediksinya.

2. **Kualitas Preprocessing Sangat Berpengaruh:** Proses crop dan resize wajah menjadi 100×100 piksel menggunakan Haar Cascade sangat krusial. Dari total dataset, 2 gambar gagal dideteksi wajahnya dan harus dihapus dari training data, membuktikan bahwa kualitas preprocessing menentukan performa model secara langsung.

3. **Label Encoding sebagai Jembatan Data:** Konversi nama string ke angka integer melalui label encoding adalah langkah wajib karena algoritma machine learning hanya dapat memproses data numerik. Tanpa langkah ini, proses training tidak bisa berjalan.

4. **Realtime Feasibility:** Implementasi real-time pada webcam berjalan dengan lancar. Loop membaca frame, mendeteksi wajah, dan memprediksi identitas terjadi dalam hitungan milidetik, membuktikan bahwa LBPH cukup ringan untuk dijalankan secara real-time tanpa memerlukan GPU.

5. **Kesimpulan:** Sistem face recognition berbasis OpenCV LBPH mampu mencapai akurasi 97% dengan dataset yang relatif kecil (70 gambar per orang). Ini menjadi bukti bahwa algoritma klasik computer vision masih sangat relevan dan efisien untuk kasus pengenalan wajah dalam kondisi terkontrol.

---

## 📽️ Presentasi Video

Demonstrasi lengkap mengenai langkah-langkah praktikum, penjelasan kode baris demi baris, dan eksekusi live di VSCode dapat diakses melalui tautan berikut:

👉 [ https://youtu.be/lX8QpkStQfY]

---
