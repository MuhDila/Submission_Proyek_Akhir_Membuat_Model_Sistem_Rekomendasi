# Proyek Akhir : Membuat Model Sistem Rekomendasi
- **Nama:** Muhammad Dila
- **Email:** muhammaddila.all@gmail.com
- **ID Dicoding:** muhdila

---

# Project Overview

Membaca buku adalah aktivitas penting untuk pengembangan pengetahuan, keterampilan berpikir kritis, dan kesehatan mental. Namun, di era digital saat ini, pembaca dihadapkan pada jutaan pilihan buku di platform online, yang menyebabkan fenomena "choice overload".

Menghadapi tantangan tersebut, dibutuhkan sistem rekomendasi yang dapat membantu pengguna menemukan buku-buku yang relevan dengan minat mereka berdasarkan riwayat interaksi sebelumnya (rating, ulasan).

**Mengapa masalah ini penting untuk diselesaikan?**

- Banyak pengguna merasa kesulitan menemukan buku yang sesuai dengan preferensi mereka karena terlalu banyak pilihan yang tersedia.
- Sistem rekomendasi dapat meningkatkan engagement pengguna dan loyalitas terhadap platform buku digital.
- Membantu penerbit dan penulis untuk menjangkau audiens yang lebih spesifik dan relevan dengan genre atau kategori bukunya.

**Bagaimana cara menyelesaikannya?**

- Dengan membangun sistem rekomendasi berbasis:
  - **Content-Based Filtering** (menggunakan kemiripan judul buku melalui TF-IDF dan Cosine Similarity),
  - **User-Based Collaborative Filtering (Memory-Based)** (menggunakan pola rating antar pengguna),
  - **Model-Based Collaborative Filtering** (menggunakan embedding learning dengan TensorFlow/Keras).
- Dataset yang digunakan adalah [Book Recommendation Dataset dari Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset).

**Referensi:**

- Alharthi, A., Zaki, M., & Alshamrani, S. (2018). "A Survey of Book Recommender Systems", *Journal of Intelligent Information Systems*. [Link](https://link.springer.com/article/10.1007/s10844-017-0489-9)

---

# Business Understanding

Pada bagian ini, kami akan menjelaskan proses klarifikasi masalah yang menjadi dasar dalam membangun sistem rekomendasi buku berbasis machine learning yang mencakup:

---

## Problem Statements

- Dengan banyaknya pilihan buku di platform online, pengguna mengalami kesulitan menemukan buku yang sesuai dengan preferensi mereka.
- Banyak pengguna baru atau pengguna pasif tidak memberikan cukup rating, sehingga sulit memahami minat mereka tanpa bantuan sistem rekomendasi.
- Platform perlu meningkatkan pengalaman pengguna agar lebih personal, untuk meningkatkan keterlibatan (engagement) dan loyalitas pengguna terhadap aplikasi atau layanan buku digital.

---

## Goals

- Membantu pengguna menemukan buku yang relevan dengan minat mereka berdasarkan interaksi sebelumnya (rating yang diberikan).
- Mengurangi fenomena "choice overload" dengan menyediakan rekomendasi top-N buku secara personal.
- Meningkatkan tingkat engagement dan waktu yang dihabiskan pengguna dalam platform melalui sistem rekomendasi yang efektif.
- Mendorong pengguna untuk mengeksplorasi lebih banyak buku dari kategori atau genre yang mungkin mereka sukai.

---

## Solution Statements

- **Content-Based Filtering (CBF)**  
  Menggunakan informasi konten buku, seperti *judul* dan *kemiripan antar judul* (berbasis TF-IDF dan Cosine Similarity), untuk merekomendasikan buku lain yang serupa dengan buku yang pernah disukai pengguna.  
  *Kelebihan:* Dapat memberikan rekomendasi meskipun user baru dan minim interaksi.  
  *Kekurangan:* Bisa terbatas pada buku-buku serupa saja (kurang eksplorasi).

- **User-Based Collaborative Filtering (Memory-Based)**  
  Menggunakan pola rating antar pengguna untuk merekomendasikan buku berdasarkan kemiripan preferensi antar user.  
  *Kelebihan:* Lebih mampu menangkap pola selera pengguna yang beragam.  
  *Kekurangan:* Mengalami masalah cold-start jika user baru atau rating sedikit.

- **Model-Based Collaborative Filtering (Deep Learning dengan Keras)**  
  Menggunakan teknik embedding pada user dan buku untuk mempelajari representasi laten preferensi, sehingga dapat memberikan prediksi rating dan rekomendasi yang lebih akurat.  
  *Kelebihan:* Skalabilitas tinggi, mampu menangkap pola kompleks di data.  
  *Kekurangan:* Membutuhkan lebih banyak data dan proses training lebih lama dibanding metode sederhana.

---

# Data Understanding

Pada bagian ini, kami menjelaskan jumlah data, kondisi data, dan informasi mengenai dataset yang digunakan. Dataset yang digunakan dalam proyek ini adalah [Book Recommendation Dataset - Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset).

Dataset ini terdiri dari tiga file utama:
- `Books.csv` : berisi informasi mengenai buku, seperti ISBN, judul, penulis, tahun terbit, penerbit, dan gambar sampul.
- `Users.csv` : berisi informasi mengenai pengguna, seperti User-ID, lokasi, dan usia.
- `Ratings.csv` : berisi data interaksi antara pengguna dan buku, berupa rating yang diberikan.

---

## Informasi Jumlah Data

- Books.csv: 271.360 entri, 8 kolom
- Users.csv: 278.858 entri, 3 kolom
- Ratings.csv: 1.149.780 entri, 3 kolom

---

## Uraian Variabel

**Books.csv**
- `ISBN` : ID unik buku. Tipe data **object**. Jumlah non-null: **271.360**.
- `Book-Title` : Judul buku. Tipe data **object**. Jumlah non-null: **271.360**.
- `Book-Author` : Nama penulis buku. Tipe data **object**. Jumlah non-null: **271.358** (terdapat 2 missing value).
- `Year-Of-Publication` : Tahun terbit buku. Tipe data **object**. Jumlah non-null: **271.360**.
- `Publisher` : Nama penerbit buku. Tipe data **object**. Jumlah non-null: **271.358** (terdapat 2 missing value).
- `Image-URL-S` : URL gambar sampul kecil. Tipe data **object**. Jumlah non-null: **271.360**.
- `Image-URL-M` : URL gambar sampul sedang. Tipe data **object**. Jumlah non-null: **271.360**.
- `Image-URL-L` : URL gambar sampul besar. Tipe data **object**. Jumlah non-null: **271.357** (terdapat 3 missing value).

**Users.csv**
- `User-ID` : ID unik pengguna. Tipe data **int64**. Jumlah non-null: **168.096**.
- `Location` : Lokasi pengguna (Kota, Provinsi, Negara). Tipe data **object**. Jumlah non-null: **168.096**.
- `Age` : Usia pengguna. Tipe data **float64**. Jumlah non-null: **168.096**.

**Ratings.csv**
- `User-ID` : ID pengguna yang memberikan rating. Tipe data **int64**. Jumlah non-null: **1.149.780**.
- `ISBN` : ISBN buku yang diberi rating. Tipe data **object**. Jumlah non-null: **1.149.780**.
- `Book-Rating` : Rating dari pengguna ke buku (skala 0–10). Tipe data **int64**. Jumlah non-null: **1.149.780**.

---

## Statistik Ringkasan

**Statistik Ringkasan Users**

Beberapa insight penting:
- Rata-rata usia pengguna adalah **34.75 tahun**.
- Usia minimum adalah **0 tahun**, dan maksimum tercatat **244 tahun**, yang kemungkinan merupakan data tidak valid.
- Lokasi paling sering muncul adalah **"london, england, united kingdom"**, dengan sebanyak **2.506 pengguna** berasal dari lokasi tersebut.
- Terdapat **57.339 lokasi unik** dalam dataset.

**Statistik Ringkasan Ratings**

Beberapa insight penting:
- Rata-rata nilai rating adalah **2.87**, dengan standar deviasi sebesar **3.85**.
- Rating minimum adalah **0** dan maksimum **10**.
- Sebanyak 50% data memiliki rating **0**, yang menunjukkan banyak user melakukan rating kosong (default).
- ISBN paling sering muncul adalah `0971880107`, sebanyak **2.502 kali**.
- Terdapat **340.556 ISBN unik** di dalam dataset.

**Statistik Ringkasan Books**

Beberapa insight penting:
- Terdapat **271.360** data ISBN yang unik, menunjukkan setiap buku memiliki ISBN berbeda.
- Terdapat **242.135** judul buku yang unik.
- Penulis paling populer dalam dataset adalah **Agatha Christie**, dengan **632** buku tercatat.
- Tahun publikasi paling sering adalah **2002**, dengan **17.627** buku diterbitkan pada tahun tersebut.
- Penerbit dengan jumlah buku terbanyak adalah **Harlequin**, sebanyak **7.535** buku.
- Untuk gambar cover (`Image-URL`), sebagian besar URL unik, namun ada URL yang dipakai lebih dari sekali.

---

## Informasi Kondisi Data

- Dataset Books:
  - Memiliki beberapa missing value di kolom `Book-Author`, `Publisher`, dan `Image-URL-L`.
- Dataset Users:
  - Terdapat sekitar 110.762 missing value pada kolom `Age`.
  - Ditemukan usia tidak valid, seperti 0 dan lebih dari 100 tahun.
- Dataset Ratings:
  - Tidak terdapat missing value.
  - Banyak pengguna memberikan rating 0.

---

## Cek Duplikat Data

Pada tahap ini, kami memeriksa apakah terdapat data duplikat pada masing-masing dataset.

Hasil pemeriksaan:
- Dataset Books: 0 duplikat
- Dataset Users: 0 duplikat
- Dataset Ratings: 0 duplikat

Tidak ditemukan data duplikat, sehingga tidak diperlukan tindakan pembersihan untuk duplikasi.

---

## Exploratory Data Analysis (EDA)

**Distribusi Usia Pengguna**

Visualisasi menunjukkan mayoritas pengguna berusia antara 20 hingga 50 tahun. Terdapat anomali pada usia 0 dan di atas 100 tahun, yang mengindikasikan adanya outlier.

![Distribusi Usia Pengguna](image/Distribusi%20Usia%20Pengguna.png)

**Insight**:
Mayoritas pengguna berada di rentang usia 30-40 tahun. Outlier usia di atas 100 tahun perlu dibersihkan pada tahap data preparation.

**Distribusi Nilai Rating**

Distribusi nilai rating menunjukkan bahwa sebagian besar rating yang diberikan adalah 0.
Rating tinggi seperti 8–10 juga cukup banyak, menunjukkan adanya bias positif pengguna terhadap buku favorit mereka.

![Distribusi Nilai Rating Buku](image/Distribusi%20Nilai%20Rating%20Buku.png)

**Insight**:
Proporsi rating 0 sangat dominan, menunjukkan banyak rating kosong atau default. Ini akan berpengaruh dalam tahap modeling sistem rekomendasi.

**10 Buku dengan Rating Terbanyak**

Berikut 10 buku dengan jumlah rating terbanyak di dataset:

- Wild Animus (2502 rating)
- The Lovely Bones: A Novel (1295 rating)
- The Da Vinci Code (883 rating)
- Divine Secrets of the Ya-Ya Sisterhood: A Novel (732 rating)
- The Red Tent (723 rating)
- A Painted House (647 rating)
- [NaN] (639 rating) — Tidak diketahui judul buku
- The Secret Life of Bees (615 rating)
- Snow Falling on Cedars (614 rating)
- Angels & Demons (586 rating)

![10 Buku dengan Rating Terbanyak](image/Buku%20dengan%20Rating%20Terbanyak.png)

**Insight**:
Buku Wild Animus jauh lebih sering dirating dibandingkan buku lain, kemungkinan besar karena faktor popularitas atau faktor marketing tertentu.

**Catatan**: Terdapat ISBN yang tidak memiliki `Book-Title` (NaN), perlu perhatian lebih dalam tahap berikutnya.

---

# Data Preparation

Pada bagian ini, kami menerapkan dan menyebutkan teknik data preparation yang dilakukan.
Teknik yang digunakan pada notebook dan laporan disusun secara berurutan sesuai proses.

---

## Handling Missing Value pada Kolom Age

**Teknik yang digunakan**:
- Menghapus baris yang memiliki missing value pada kolom `Age` menggunakan fungsi `dropna(subset=['Age'])`.

**Proses dan Alasan**:
- Ditemukan sekitar 110.762 missing value pada kolom `Age`.
- Menghapus missing value dipilih untuk menghindari bias dalam analisis usia pengguna.
- Setelah penghapusan, jumlah data Users berkurang menjadi 168.096 entri.

---

## Handling Outlier pada Kolom Age

**Teknik yang digunakan**:
- Filtering nilai `Age` untuk hanya mempertahankan data dengan usia antara 5 hingga 100 tahun.

**Proses dan Alasan**:
- Terdapat nilai usia tidak realistis seperti 0 tahun dan 244 tahun.
- Data dengan usia <5 atau >100 dihapus agar lebih representatif.
- Setelah filtering, jumlah data Users menjadi 166.848 entri.

---

## Filtering Data Rating

**Teknik yang digunakan**:
- Menghapus data rating dengan nilai `Book-Rating == 0`.

**Proses dan Alasan**:
- Rating 0 dianggap sebagai indikasi tidak memberikan rating aktif.
- Fokus analisis hanya pada rating aktif (1–10).
- Setelah filtering, jumlah data Ratings berkurang menjadi 433.671 entri.

---

## Filtering User dan Buku Berdasarkan Aktivitas Minimum

**Teknik yang digunakan**:
- Filtering untuk hanya mempertahankan:
  - User yang memberikan minimal 3 rating
  - Buku yang menerima minimal 3 rating

**Proses dan Alasan**:
- Memastikan pengguna yang dianalisis aktif dalam memberikan rating.
- Memastikan buku yang dianalisis memiliki cukup banyak feedback.
- Setelah filtering, jumlah data Ratings menjadi 203.851 entri.

---

## Membuat DataFrame Clean untuk Modeling

**Teknik yang digunakan**:
- Mengubah kolom `User-ID`, `ISBN`, dan `Book-Rating` menjadi list menggunakan `.tolist()`.
- Membentuk DataFrame baru `ratings_clean` dari list tersebut.

**Proses dan Alasan**:
- Konversi ke bentuk DataFrame clean bertujuan memudahkan proses modeling sistem rekomendasi.
- Data `ratings_clean` terdiri dari 203.851 baris dan 3 kolom: `user_id`, `isbn`, dan `book_rating`.

---

# Modeling

Pada bagian ini, kami membangun dua pendekatan sistem rekomendasi untuk menyelesaikan permasalahan prediksi buku yang relevan untuk pengguna.

---

## Content-Based Filtering

### Proses:
- Membuat TF-IDF vectorizer dari kolom `Book-Title`.
- Menghitung cosine similarity antar judul buku.
- Membuat fungsi `recommend_books(book_title)` untuk menghasilkan Top-N rekomendasi.

### Output:
- TF-IDF Matrix Shape: **(24.253, 16.052)**
- Cosine Similarity Matrix Shape: **(24.253, 24.253)**

> Matriks cosine similarity berhasil dibuat untuk mengukur kemiripan antar buku berdasarkan judul.

---

## Collaborative Filtering (Memory-Based - User-Based)

### Proses:
- Membentuk User-Item Matrix dari data rating.
- Menghitung cosine similarity antar user berdasarkan User-Item Matrix.
- Membuat fungsi `recommend_books_userbased(user_id)` untuk menghasilkan Top-N rekomendasi berdasarkan user similarity.

### Output:
- User-Item Matrix Shape: **(20.908, 25.790)**
- User Similarity Matrix Shape: **(20.908, 20.908)**

> User-item interaction berhasil dibentuk, dan kemiripan antar pengguna berhasil dihitung menggunakan cosine similarity.

---

## Collaborative Filtering (Model-Based dengan TensorFlow Keras)

### Proses:
- Encode User-ID dan ISBN ke indeks numerik.
- Membuat class custom `RecommenderNet` menggunakan Embedding Layer.
- Membagi data ke dalam training set dan validation set (rasio 80:20).
- Melatih model menggunakan loss binary crossentropy dan optimizer Adam.

### Output:
- Jumlah user unik: **20.908**
- Jumlah buku unik: **25.790**
- Shape `x_train`: **(163.080, 2)**
- Shape `y_train`: **(163.080,)**
- Shape `x_val`: **(40.771, 2)**
- Shape `y_val`: **(40.771,)**

**Hasil Training:**
- RMSE Training terbaik: sekitar **0.1517**
- RMSE Validation terbaik: sekitar **0.1835**

> Model berbasis embedding berhasil dibangun dan dilatih untuk mempelajari representasi laten user dan buku.

---

## Top-N Recommendation Output

Pada setiap pendekatan, sistem menghasilkan rekomendasi **Top-N** buku:

- Pada **Content-Based Filtering**, rekomendasi buku-buku yang memiliki kemiripan tinggi berdasarkan judul.
- Pada **Collaborative Filtering**, rekomendasi buku berdasarkan pola interaksi pengguna.

---

## Perbandingan Pendekatan

| Pendekatan | Kelebihan | Kekurangan |
|:---|:---|:---|
| **Content-Based Filtering** | - Tidak memerlukan data rating.<br>- Dapat merekomendasikan item baru. | - Hanya merekomendasikan item mirip.<br>- Tidak menangkap pola komunitas. |
| **Collaborative Filtering** | - Menangkap pola komunitas pengguna.<br>- Rekomendasi bisa lebih beragam. | - Membutuhkan data interaksi user-item.<br>- Rentan cold-start untuk user/buku baru. |

Dengan menggabungkan dua pendekatan ini, sistem rekomendasi menjadi lebih kuat dalam memberikan rekomendasi buku yang relevan baik berdasarkan konten maupun preferensi pengguna.

---

# Evaluation
Pada bagian ini, kami mengevaluasi kinerja sistem rekomendasi yang telah dibangun menggunakan dua pendekatan: Content-Based Filtering dan Collaborative Filtering (baik Memory-Based maupun Model-Based).

---

## Metrik Evaluasi: Root Mean Squared Error (RMSE)

Untuk mengukur kinerja model Collaborative Filtering berbasis TensorFlow Keras, kami menggunakan metrik **Root Mean Squared Error (RMSE)**.
RMSE banyak digunakan dalam masalah regresi, termasuk prediksi rating, karena mengukur seberapa jauh prediksi model dari nilai aktual.

**Alasan memilih RMSE:**
- Dataset berisi data **rating** yang bersifat kontinu (bukan klasifikasi).
- RMSE mampu memberikan gambaran langsung tentang besarnya error prediksi dalam satuan yang sama dengan rating.
- Semakin kecil nilai RMSE, semakin baik performa model.

---

## Hasil Evaluasi Model

- **RMSE pada Data Training**: sekitar **0.1517**
- **RMSE pada Data Validation**: sekitar **0.1835**

**Interpretasi:**
- RMSE training dan validation relatif kecil, menunjukkan bahwa model mampu melakukan prediksi rating dengan akurasi yang baik.
- Perbedaan antara RMSE training dan validation juga tidak terlalu besar, sehingga tidak terdapat indikasi overfitting yang parah.

---

## Visualisasi Learning Curve

![Learning Curve](image/Learning%20Curve%20-%20RMSE.png)

**Insight:**
- Learning curve menunjukkan bahwa model cepat konvergen dalam 10–15 epoch.
- RMSE stabil dan tidak mengalami peningkatan drastis, memperlihatkan stabilitas model dalam belajar.

---

## Evaluasi Hasil Rekomendasi

Pada tahap ini, kami menguji hasil dari masing-masing pendekatan sistem rekomendasi yang dibangun.

---

### Content-Based Filtering (CBF)

- **Input Buku:**  
  *Harry Potter and the Chamber of Secrets (Book 2)*

- **Top-5 Rekomendasi Buku:**

| No | Book-Title | Book-Author |
|:--|:--|:--|
| 1 | Harry Potter and the Chamber of Secrets (Book 2) | J.K. Rowling |
| 2 | Harry Potter and the Chamber of Secrets (Book 2) | J.K. Rowling |
| 3 | Harry Potter and the Chamber of Secrets (Harry Potter) | J.K. Rowling |
| 4 | Harry Potter and the Chamber of Secrets Postcard Book | J.K. Rowling |
| 5 | Harry Potter and the Chamber of Secrets (Book 2 Audio CD) | J.K. Rowling |

- **Interpretasi:**
  - Rekomendasi yang dihasilkan seluruhnya adalah varian dari buku Harry Potter.
  - Ini membuktikan bahwa sistem berhasil menemukan buku-buku dengan kemiripan konten yang tinggi berdasarkan judul (menggunakan TF-IDF + Cosine Similarity).
  - Efektif untuk merekomendasikan buku lain yang serupa secara tema dan edisi.

---

### User-Based Collaborative Filtering (Memory-Based)

- **Contoh User-ID:**  
  *8*

- **Top-3 Rekomendasi Buku:**

| No | Book-Title | Book-Author |
|:--|:--|:--|
| 1 | To Kill a Mockingbird | Harper Lee |
| 2 | ANGELA'S ASHES | Frank McCourt |
| 3 | Outlander | Diana Gabaldon |

- **Interpretasi:**
  - Buku-buku yang direkomendasikan adalah novel fiksi terkenal dengan reputasi kuat.
  - Model berhasil menangkap pola preferensi user berdasarkan perilaku user-user lain yang mirip.
  - Menunjukkan bahwa user 8 memiliki ketertarikan terhadap kategori fiksi populer dan historical novel.

---

### Model-Based Collaborative Filtering (Keras)

- **Contoh User-ID:**  
  *263663*

- **Top-5 Rekomendasi Buku:**

| No | Book-Title | Book-Author |
|:--|:--|:--|
| 1 | The Return of the King (The Lord of the Rings, Part 3) | J.R.R. Tolkien |
| 2 | The Two Towers (The Lord of the Rings, Part 2) | J.R.R. Tolkien |
| 3 | The Giving Tree | Shel Silverstein |
| 4 | Dilbert: A Book of Postcards | Scott Adams |
| 5 | Harry Potter and the Chamber of Secrets Postcard Book | J.K. Rowling |

- **Interpretasi:**
  - Model merekomendasikan buku-buku fantasy dan karya klasik populer.
  - Tolkien dan Rowling mendominasi rekomendasi, menunjukkan kecenderungan minat user terhadap genre fantasi dan literatur klasik.
  - Buku humor seperti *Dilbert* menunjukkan model juga menangkap diversifikasi minat user.
  - Ini membuktikan bahwa model embedding berbasis Keras mampu merepresentasikan preferensi user secara laten dan memberikan prediksi rekomendasi personal yang bermakna.

---

## Kesimpulan Evaluasi Hasil Rekomendasi

- Content-Based Filtering efektif dalam menemukan buku-buku serupa berdasarkan fitur konten.
- User-Based Collaborative Filtering efektif menghubungkan user dengan user lain yang memiliki minat serupa.
- Model-Based Collaborative Filtering berbasis Keras mampu menangkap pola kompleks dalam interaksi user dan buku, dan menghasilkan rekomendasi yang lebih fleksibel dan personal.

**Semua metrik dan hasil evaluasi konsisten dengan konteks data, problem statement, dan solusi yang ditargetkan.**