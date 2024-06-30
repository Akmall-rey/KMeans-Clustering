import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns

def main():
    # Insert containers separated into tabs:
    tab1, tab2, tab3 = st.tabs(["Artikel", "Analysis", "Our Team"])

    # You can also use "with" notation:
    with tab1:

        st.markdown("""
            <style>
            .title {
                text-align: center;
            }
            .justified {
                text-align: justify;
            }
            .center-img {
                display: flex;
                justify-content: center;
                margin: 20px 0;
            }
            .center-text {
                display: flex;
                justify-content: center;
            }
            .bold {
                font-weight: bold;
            }
            </style>
            <h2 class="title">ANALISIS KESEHATAN BALITA UNTUK SKALA PRIORITAS PROGRAM PENANGGULANGAN DI KECAMATAN BOYOLALI DENGAN K-MEANS CLUSTERING</h2>
            <div class="justified">
                <div class="center-img">
                    <img src="https://pyfahealth.com/wp-content/uploads/2022/12/Gejala-Pneumonia-pada-Anak-dan-Cara-Mengobati-.jpg" alt="Deskripsi gambar" width="500">
                </div>
                <h3>Permasalahan</h3>
                    <p>Kesehatan suatu negara dillihat dari beberapa indikator utama salah satunya yaitu kesehatan balita. Masalah kesehatan balita secara umum terdiri dari pneumonia, diare, gizi kurang, dan gizi buruk. Pemerintah sudah melakukan penanganan terhadap masalah kesehatan balita ini tetapi penanganan tersebut seringkali tidak efisien dan tepat sasaran. Solusi nya adalah dengan mengelompokkan daerah yang memiliki kemiripan data dengan menggunakan algoritma K-Means Clustering. Penelitian serupa juga pernah dilakukan di Kabupaten Bengkulu Utara dengan kesimpulan daerah yang dinilai memiliki tingkat kesehatan tinggi ternyata setelah dikelompokkan daerah tersebut termasuk ke dalam daerah dengan tingkat kesehatan rendah.</p>
                <h6 class="bold">Bagaimana Cara Menanganinya?</h6>
                    <p>Berdasarkan permasalahan yang telah dijabarkan, penerapan algoritma K-Means Clustering dirasa dapat membantu dalam menganalisis data kesehatan balita menurut kecamatan untuk memprioritaskan program penanggulangan yang lebih efektif. Analisis ini mengambil beberapa parameter seperti diare, pneumonia, gizi buruk, gizi kurang, dan kematian balita, dengan tujuan untuk mengidentifikasi kecamatan-kecamatan yang memerlukan perhatian khusus dan intervensi kesehatan yang lebih intensif di Kabupaten Boyolali. Dengan demikian, program penanggulangan kesehatan balita dapat lebih terfokus dan tepat sasaran, meningkatkan kualitas hidup dan kesehatan anak-anak di wilayah tersebut.</p>
                <h3>Tahapan Penelitian</h3>
                    <p class="bold">1. Pengumpulan Data</p>
                    <p>Data dikumpulkan dari berbagai sumber yang terpercaya, seperti jurnal ilmiah, artikel dan dataset publik yang relevan. Proses pengumpulan data melibatkan pencarian dan pemilihan sumber yang sesuai dengan kriteria penelitian, serta mengumpulkan data dari sumber-sumber tersebut. Pada penelitian kali ini, penulis mengambil dataset dari web <a href="https://data.boyolali.go.id/">Portal Data Boyolali</a>.</p>
                    <p class="bold">2. Perencanaan dan Penentuan Metode</p>
                    <p>Setelah data terkumpul, langkah selanjutnya yang dilakukan adalah menentukan apa yang akan dibuat dan perencanaan tahapan-tahapan penelitian yang akan dilakukan, termasuk menentukan parameter yang digunakan untuk analisis.</p>
                    <p class="bold">3. Implementasi dan Pengujian</p>
                    <p>Pada tahap ini, dataset akan melalui proses olah data menggunakan algoritma K-Means dan diuji menggunakan aplikasi seperti RapidMiner dan Google Colab. Pengujian harus dilakukan secara menyeluruh untuk mengidentifikasi kesalahan ataupun masalah kinerja dari program yang dibuat. Proses ini melibatkan revisi dan debugging kode untuk memastikan algoritma berfungsi dengan benar dan menghasilkan output yang diharapkan.</p>
                    <p class="bold">4. Penyelesaian dan Analisis Hasil</p>
                    <p>Setelah pengujian berhasil dan output yang dihasilkan sesuai, data yang telah diolah dimasukkan ke dalam model untuk melihat hasil klasterisasi. Hasil dari klasterisasi dianalisis untuk mengidentifikasi kecamatan-kecamatan dengan kondisi kesehatan balita yang memerlukan perhatian khusus. Kesimpulan dibuat berdasarkan hasil analisis ini untuk memberikan rekomendasi program penanggulangan kesehatan yang lebih efektif di Kabupaten Boyolali.</p>
                <h3>Hasil Analisis<h3>
            </div>
            """, unsafe_allow_html=True)
        
    with tab2:

        st.header("Clustering Analysis")

        # File uploader
        st.info("Upload data excel dengan format table seperti contoh dibawah")
        st.markdown("""
        <table>
            <tr>
                <th>KECAMATAN</th>
                <th>BALITA PENDERITA DIARE</th>
                <th>BALITA PENDERITA PNEUMONIA</th>
                <th>BALITA GIZI KURANG</th>
                <th>BALITA GIZI BURUK</th>
                <th>KEMATIAN BALITA</th>
            </tr>
            <tr>
                <td>Jebres</td>
                <td>1</td>
                <td>2</td>
                <td>3</td>
                <td>4</td>
                <td>5</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

        if uploaded_file is not None:
            data = pd.read_excel(uploaded_file)

            st.write("Dataset:")
            st.table(data)

            # Menentukan Kategori
            statistics = data.describe().T
            statistics['mean'] = statistics['mean']
            statistics['std_dev'] = statistics['std']
            statistics['low'] = statistics['mean'] - statistics['std_dev']
            statistics['high'] = statistics['mean'] + statistics['std_dev']

            st.header("Proses Kategorisasi")
            st.write("Statistik Deskriptif:")
            st.write(statistics[['mean', 'std_dev', 'low', 'high']])

            # Fungsi Kategori
            def categorize(value, low, high):
                if value < low:
                    return 'rendah'
                elif value > high:
                    return 'tinggi'
                else:
                    return 'sedang'

            for column in data.columns:
                if column not in ['KECAMATAN']:
                    low = statistics.loc[column, 'low']
                    high = statistics.loc[column, 'high']
                    data[column + ' KATEGORI'] = data[column].apply(categorize, args=(low, high))

            st.write("Data dengan kategori:")
            st.table(data)

            # Normalisasi Data
            dataCluster = data.iloc[:, 1:6].fillna(0)
            scaler = MinMaxScaler()
            normalisasiData = scaler.fit_transform(dataCluster)

            normalized_df = pd.DataFrame(normalisasiData, columns=data.columns[1:6])
            normalized_df['KECAMATAN'] = data['KECAMATAN'].values

            normalized_df = normalized_df[['KECAMATAN'] + list(normalized_df.columns[:-1])]

            st.write("Data yang sudah dinormalisasi:")
            st.table(normalized_df)


            # PCA sebelum klaster
            pca = PCA(n_components=5)
            pca_data = pca.fit_transform(normalisasiData)
            pca_data_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
            pca_data_df['Kecamatan'] = data.iloc[:, 0]

            plt.figure(figsize=(10, 6))
            kecamatan = pca_data_df['Kecamatan'].unique()

            for kec in kecamatan:
                condition_data = pca_data_df[pca_data_df['Kecamatan'] == kec]
                plt.scatter(condition_data['PC1'], condition_data['PC2'], label=kec)

            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('Scatter Plot PCA using PC1 and PC2')
            plt.grid(True)

            st.write("Visualisasi dengan PCA:")
            st.pyplot(plt)

            # Menentukan rentang nilai k
            n_samples = normalisasiData.shape[0]
            k_range = range(2, min(11, n_samples))

            silhouette_scores = []

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, max_iter=300, n_init=10, init='k-means++')
                cluster_labels = kmeans.fit_predict(normalisasiData)
                silhouette_avg = silhouette_score(normalisasiData, cluster_labels)
                silhouette_scores.append(silhouette_avg)

            # Plotting
            st.write("")
            st.header("Proses Clustering")
            st.write("Mencari nilai k optimal")
            plt.figure(figsize=(10, 6))
            plt.plot(k_range, silhouette_scores, marker='o')
            plt.xlabel('Jumlah Klaster (k)')
            plt.ylabel('Rata-rata Silhouette Score')
            plt.title('Silhouette Method')
            plt.grid(True)

            st.pyplot(plt)
            plt.close()

            # Menentukan nilai k
            optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
            st.write(f"Didapatkan nilai k optimal: {optimal_k}")
            st.write("")

            # Kmeans Clustering
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, max_iter=300, n_init=10, init='k-means++')
            kmeans.fit(normalisasiData)

            # Menampilkan centroid awal
            st.write("Centroid Awal:")
            initial_centroids = kmeans.cluster_centers_
            st.write(pd.DataFrame(initial_centroids, columns=data.columns[1:6]))

            cluster_labels = kmeans.predict(normalisasiData)
            data['Cluster'] = cluster_labels

            # Menampilkan anggota dari setiap klaster dalam tabel
            for cluster in range(optimal_k):
                st.write(f"Anggota dari Cluster {cluster}:")
                cluster_members = data[data['Cluster'] == cluster]
                st.write(cluster_members)

            # PCA setelah klaster
            pca = PCA(n_components=5)
            pca_data = pca.fit_transform(normalisasiData)
            pca_data_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
            pca_data_df['Cluster'] = cluster_labels

            # Plotting
            plt.figure(figsize=(10, 6))
            palette = sns.color_palette('hsv', optimal_k)
            for cluster in range(optimal_k):
                cluster_data = pca_data_df[pca_data_df['Cluster'] == cluster]
                plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {cluster}', color=palette[cluster])
            plt.xlabel('PC 1')
            plt.ylabel('PC 2')
            plt.title('Cluster Plot with PCA (PC1 and PC2)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            st.pyplot(plt)
            plt.close()

            # Cluster categories
            cluster_categories = {
                'Cluster': [],
                'Rendah': [],
                'Sedang': [],
                'Tinggi': []
            }
            for cluster in range(optimal_k):
                cluster_data = data[data['Cluster'] == cluster]
                low_count = cluster_data.filter(like='KATEGORI').apply(lambda x: (x == 'rendah').sum()).sum()
                medium_count = cluster_data.filter(like='KATEGORI').apply(lambda x: (x == 'sedang').sum()).sum()
                high_count = cluster_data.filter(like='KATEGORI').apply(lambda x: (x == 'tinggi').sum()).sum()
                cluster_categories['Cluster'].append(cluster)
                cluster_categories['Rendah'].append(low_count)
                cluster_categories['Sedang'].append(medium_count)
                cluster_categories['Tinggi'].append(high_count)

            df_cluster_categories = pd.DataFrame(cluster_categories)

            # Menambahkan kolom prioritas
            def calculate_priority(row):
                # Skor Indeks
                rendah_bobot = 1
                sedang_bobot = 2
                tinggi_bobot = 3

                return (row['Rendah'] * rendah_bobot + row['Sedang'] * sedang_bobot + row['Tinggi'] * tinggi_bobot) / (row['Rendah'] + row['Sedang'] + row['Tinggi'])

            df_cluster_categories['Prioritas'] = df_cluster_categories.apply(calculate_priority, axis=1)

            st.write("Cluster Prioritas:")
            st.write(df_cluster_categories)

    with tab3:
        col1, col2, col3 = st.columns(3, gap = "medium")
        
        
        col1.image("img/Poltak.jpg")
        col1.write("Poltak Alfredo Sitorus Philander")
        col1.write("L0122125")
        col2.image("img/Akmal.jpg")
        col2.write("Raihan Akmal Darmawan")
        col2.write("L0122131")
        col3.image("img/Pian.jpg")
        col3.write("Raihan Havian")
        col3.write("L0122132")


if __name__ == "__main__":
    main()
