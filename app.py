import streamlit as st
import h5py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Keranjang Belanja",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

class H5MarketBasketRecommender:
    """
    Sistem Rekomendasi Keranjang Belanja yang menggunakan model H5
    Kompatibel dengan arsitektur TensorFlow/Keras untuk pipeline ML
    """
    
    def __init__(self, h5_model_path):
        """Memuat model dari file H5"""
        self.model_path = h5_model_path
        self.load_model()
    
    def load_model(self):
        """Memuat semua komponen model dari file H5"""
        with h5py.File(self.model_path, 'r') as f:
            
            # Memuat metadata
            self.metadata = dict(f['model_metadata'].attrs)
            
            # Memuat frequent itemsets
            if 'frequent_itemsets' in f:
                freq_group = f['frequent_itemsets']
                self.frequent_itemsets = {
                    'itemsets': [eval(s.decode('utf-8')) for s in freq_group['itemsets'][:]],
                    'support': freq_group['support'][:],
                    'length': freq_group['length'][:]
                }
            
            # Memuat aturan asosiasi
            if 'association_rules' in f:
                rules_group = f['association_rules']
                self.association_rules = {
                    'antecedents': [eval(s.decode('utf-8')) for s in rules_group['antecedents'][:]],
                    'consequents': [eval(s.decode('utf-8')) for s in rules_group['consequents'][:]],
                    'support': rules_group['support'][:],
                    'confidence': rules_group['confidence'][:],
                    'lift': rules_group['lift'][:]
                }
            
            # Memuat aturan cross-selling
            if 'cross_selling_rules' in f and len(f['cross_selling_rules'].keys()) > 0:
                cross_group = f['cross_selling_rules']
                self.cross_selling_rules = {
                    'antecedents': [eval(s.decode('utf-8')) for s in cross_group['antecedents'][:]],
                    'consequents': [eval(s.decode('utf-8')) for s in cross_group['consequents'][:]],
                    'support': cross_group['support'][:],
                    'confidence': cross_group['confidence'][:],
                    'lift': cross_group['lift'][:]
                }
            else:
                self.cross_selling_rules = None
            
            # Memuat aturan upselling
            if 'upselling_rules' in f and len(f['upselling_rules'].keys()) > 0:
                up_group = f['upselling_rules']
                self.upselling_rules = {
                    'antecedents': [eval(s.decode('utf-8')) for s in up_group['antecedents'][:]],
                    'consequents': [eval(s.decode('utf-8')) for s in up_group['consequents'][:]],
                    'support': up_group['support'][:],
                    'confidence': up_group['confidence'][:],
                    'lift': up_group['lift'][:]
                }
            else:
                self.upselling_rules = None
            
            # Memuat statistik item
            if 'item_statistics' in f:
                items_group = f['item_statistics']
                self.item_names = [s.decode('utf-8') for s in items_group['item_names'][:]]
                self.item_frequencies = items_group['item_frequencies'][:]
                self.all_items = [s.decode('utf-8') for s in items_group['all_items'][:]]
            
            # Memuat metrik performa
            if 'performance_metrics' in f:
                self.performance_metrics = dict(f['performance_metrics'].attrs)
    
    def get_cross_selling_recommendations(self, item, top_n=5, min_confidence=0.3):
        """Mendapatkan rekomendasi cross-selling untuk item tertentu"""
        if not self.cross_selling_rules:
            return []
        
        recommendations = []
        
        for i, antecedent in enumerate(self.cross_selling_rules['antecedents']):
            if item in antecedent and self.cross_selling_rules['confidence'][i] >= min_confidence:
                consequent = self.cross_selling_rules['consequents'][i]
                if len(consequent) == 1:  # Cross-selling: 1 -> 1
                    recommendations.append({
                        'item_rekomendasi': consequent[0],
                        'tingkat_kepercayaan': float(self.cross_selling_rules['confidence'][i]),
                        'dukungan': float(self.cross_selling_rules['support'][i]),
                        'lift': float(self.cross_selling_rules['lift'][i]),
                        'kekuatan_aturan': 'cross_selling'
                    })
        
        # Urutkan berdasarkan confidence dan kembalikan top N
        recommendations.sort(key=lambda x: x['tingkat_kepercayaan'], reverse=True)
        return recommendations[:top_n]
    
    def get_upselling_recommendations(self, item, top_n=5, min_confidence=0.25):
        """Mendapatkan rekomendasi upselling untuk item tertentu"""
        if not self.upselling_rules:
            return []
        
        recommendations = []
        
        for i, antecedent in enumerate(self.upselling_rules['antecedents']):
            if item in antecedent and self.upselling_rules['confidence'][i] >= min_confidence:
                consequents = self.upselling_rules['consequents'][i]
                recommendations.append({
                    'item_rekomendasi': consequents,
                    'tingkat_kepercayaan': float(self.upselling_rules['confidence'][i]),
                    'dukungan': float(self.upselling_rules['support'][i]),
                    'lift': float(self.upselling_rules['lift'][i]),
                    'kekuatan_aturan': 'upselling',
                    'ukuran_paket': len(consequents)
                })
        
        # Urutkan berdasarkan confidence dan kembalikan top N
        recommendations.sort(key=lambda x: x['tingkat_kepercayaan'], reverse=True)
        return recommendations[:top_n]
    
    def get_basket_recommendations(self, basket_items, top_n=10, min_confidence=0.2):
        """Rekomendasi berdasarkan keranjang belanja saat ini"""
        all_recommendations = {}
        
        for item in basket_items:
            if item in self.all_items:
                # Dapatkan rekomendasi cross-selling
                cross_recs = self.get_cross_selling_recommendations(item, top_n=20, min_confidence=min_confidence)
                
                for rec in cross_recs:
                    rec_item = rec['item_rekomendasi']
                    if rec_item not in basket_items:  # Jangan rekomendasikan item yang sudah ada
                        if rec_item not in all_recommendations:
                            all_recommendations[rec_item] = {
                                'total_kepercayaan': 0,
                                'total_dukungan': 0,
                                'total_lift': 0,
                                'jumlah_aturan': 0,
                                'item_pendukung': []
                            }
                        
                        all_recommendations[rec_item]['total_kepercayaan'] += rec['tingkat_kepercayaan']
                        all_recommendations[rec_item]['total_dukungan'] += rec['dukungan']
                        all_recommendations[rec_item]['total_lift'] += rec['lift']
                        all_recommendations[rec_item]['jumlah_aturan'] += 1
                        all_recommendations[rec_item]['item_pendukung'].append(item)
        
        # Hitung rata-rata dan buat rekomendasi final
        final_recommendations = []
        for item, stats in all_recommendations.items():
            if stats['jumlah_aturan'] > 0:
                final_recommendations.append({
                    'item_rekomendasi': item,
                    'rata_kepercayaan': stats['total_kepercayaan'] / stats['jumlah_aturan'],
                    'rata_dukungan': stats['total_dukungan'] / stats['jumlah_aturan'],
                    'rata_lift': stats['total_lift'] / stats['jumlah_aturan'],
                    'aturan_pendukung': stats['jumlah_aturan'],
                    'item_pendukung': stats['item_pendukung']
                })
        
        # Urutkan berdasarkan rata-rata confidence
        final_recommendations.sort(key=lambda x: x['rata_kepercayaan'], reverse=True)
        return final_recommendations[:top_n]
    
    def get_model_info(self):
        """Informasi model dan performa"""
        return {
            'metadata_model': self.metadata,
            'metrik_performa': self.performance_metrics if hasattr(self, 'performance_metrics') else {},
            'total_item': len(self.all_items) if hasattr(self, 'all_items') else 0,
            'item_teratas': dict(zip(self.item_names[:10], self.item_frequencies[:10])) if hasattr(self, 'item_names') else {}
        }

@st.cache_resource
def load_model(model_path):
    """Cache pemuatan model untuk meningkatkan performa"""
    try:
        return H5MarketBasketRecommender(model_path)
    except Exception as e:
        st.error(f"Kesalahan saat memuat model: {str(e)}")
        return None

def main():
    # Judul dan Header
    st.title("🛒 Dashboard Analisis Keranjang Belanja")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📊 Navigasi")
    
    # Input path model
    model_path = st.sidebar.text_input(
        "Path File Model H5", 
        value="market_basket_model.h5",
        help="Masukkan lokasi file model H5 Anda"
    )
    
    # Muat model
    if st.sidebar.button("🔄 Muat Model"):
        st.session_state.model = load_model(model_path)
        if st.session_state.model:
            st.sidebar.success("✅ Model berhasil dimuat!")
        else:
            st.sidebar.error("❌ Gagal memuat model")
    
    # Inisialisasi model jika belum ada
    if 'model' not in st.session_state:
        st.session_state.model = load_model(model_path)
    
    if st.session_state.model is None:
        st.error("⚠️ Pastikan file model tersedia dan coba muat ulang.")
        return
    
    model = st.session_state.model
    
    # Navigasi
    page = st.sidebar.selectbox(
        "Pilih Jenis Analisis",
        ["📈 Ringkasan Model", "🎯 Rekomendasi Item Tunggal", "🛍️ Rekomendasi Keranjang", "📊 Analitik Lanjutan"]
    )
    
    if page == "📈 Ringkasan Model":
        show_model_overview(model)
    elif page == "🎯 Rekomendasi Item Tunggal":
        show_single_item_recommendations(model)
    elif page == "🛍️ Rekomendasi Keranjang":
        show_basket_recommendations(model)
    elif page == "📊 Analitik Lanjutan":
        show_advanced_analytics(model)

def show_model_overview(model):
    """Menampilkan ringkasan model dan statistik"""
    st.header("📈 Ringkasan Model")
    
    # Dapatkan info model
    model_info = model.get_model_info()
    
    # Metrik utama dalam kolom
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Item", 
            model_info['total_item'],
            help="Jumlah item unik dalam dataset"
        )
    
    with col2:
        st.metric(
            "Total Aturan", 
            model_info['metadata_model']['total_association_rules'],
            help="Jumlah aturan asosiasi yang dihasilkan"
        )
    
    with col3:
        st.metric(
            "Total Transaksi", 
            model_info['metadata_model']['total_transactions'],
            help="Jumlah transaksi yang dianalisis"
        )
    
    with col4:
        avg_conf = model_info['metrik_performa'].get('avg_confidence', 0)
        st.metric(
            "Rata-rata Confidence", 
            f"{avg_conf:.3f}",
            help="Rata-rata confidence aturan asosiasi"
        )
    
    # Detail model
    st.subheader("🔧 Konfigurasi Model")
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.info(f"**Algoritma:** {model_info['metadata_model']['algorithm']}")
        st.info(f"**Min Support:** {model_info['metadata_model']['min_support_threshold']}")
        st.info(f"**Min Confidence:** {model_info['metadata_model']['min_confidence_threshold']}")
    
    with config_col2:
        st.info(f"**Dibuat:** {model_info['metadata_model']['created_at'][:19]}")
        st.info(f"**Framework:** {model_info['metadata_model']['framework_version']}")
        st.info(f"**Frequent Itemsets:** {model_info['metadata_model']['total_frequent_itemsets']}")
    
    # Grafik item teratas
    st.subheader("🏆 10 Item Terpopuler")
    if model_info['item_teratas']:
        items_df = pd.DataFrame(
            list(model_info['item_teratas'].items()),
            columns=['Item', 'Frekuensi']
        )
        
        fig = px.bar(
            items_df, 
            x='Frekuensi', 
            y='Item', 
            orientation='h',
            color='Frekuensi',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Metrik performa
    if 'metrik_performa' in model_info and model_info['metrik_performa']:
        st.subheader("📊 Metrik Performa")
        perf_metrics = model_info['metrik_performa']
        
        metrics_df = pd.DataFrame({
            'Metrik': ['Support', 'Confidence', 'Lift'],
            'Rata-rata': [
                perf_metrics.get('avg_support', 0),
                perf_metrics.get('avg_confidence', 0),
                perf_metrics.get('avg_lift', 0)
            ],
            'Maksimum': [
                perf_metrics.get('max_support', 0),
                perf_metrics.get('max_confidence', 0),
                perf_metrics.get('max_lift', 0)
            ],
            'Minimum': [
                perf_metrics.get('min_support', 0),
                perf_metrics.get('min_confidence', 0),
                perf_metrics.get('min_lift', 0)
            ]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Rata-rata', x=metrics_df['Metrik'], y=metrics_df['Rata-rata']))
        fig.add_trace(go.Bar(name='Maksimum', x=metrics_df['Metrik'], y=metrics_df['Maksimum']))
        fig.add_trace(go.Bar(name='Minimum', x=metrics_df['Metrik'], y=metrics_df['Minimum']))
        
        fig.update_layout(
            title="Metrik Performa Aturan Asosiasi",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def show_single_item_recommendations(model):
    """Menampilkan rekomendasi untuk item tunggal"""
    st.header("🎯 Rekomendasi Item Tunggal")
    
    # Pemilihan item
    available_items = model.all_items if hasattr(model, 'all_items') else []
    
    if not available_items:
        st.error("Tidak ada item tersedia dalam model.")
        return
    
    selected_item = st.selectbox(
        "Pilih item untuk mendapatkan rekomendasi:",
        available_items,
        help="Pilih item untuk melihat item lain yang sering dibeli bersamaan"
    )
    
    # Konfigurasi
    col1, col2 = st.columns(2)
    with col1:
        confidence_threshold = st.slider(
            "Batas Minimum Confidence", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.3, 
            step=0.05,
            help="Nilai yang lebih tinggi memberikan rekomendasi yang lebih dapat diandalkan"
        )
    
    with col2:
        top_n = st.slider(
            "Jumlah Rekomendasi", 
            min_value=1, 
            max_value=20, 
            value=10, 
            step=1
        )
    
    if st.button("🔍 Dapatkan Rekomendasi"):
        # Dapatkan rekomendasi cross-selling
        cross_recs = model.get_cross_selling_recommendations(
            selected_item, 
            top_n=top_n, 
            min_confidence=confidence_threshold
        )
        
        # Dapatkan rekomendasi upselling
        up_recs = model.get_upselling_recommendations(
            selected_item, 
            top_n=top_n, 
            min_confidence=confidence_threshold-0.05
        )
        
        # Tampilkan hasil
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔄 Rekomendasi Cross-Selling")
            if cross_recs:
                cross_df = pd.DataFrame(cross_recs)
                cross_df['tingkat_kepercayaan'] = cross_df['tingkat_kepercayaan'].round(3)
                cross_df['dukungan'] = cross_df['dukungan'].round(3)
                cross_df['lift'] = cross_df['lift'].round(3)
                
                # Ganti nama kolom untuk tampilan
                cross_df_display = cross_df.rename(columns={
                    'item_rekomendasi': 'Item Rekomendasi',
                    'tingkat_kepercayaan': 'Confidence',
                    'dukungan': 'Support',
                    'lift': 'Lift',
                    'kekuatan_aturan': 'Jenis Aturan'
                })
                
                st.dataframe(cross_df_display, use_container_width=True)
                
                # Visualisasi
                fig = px.bar(
                    cross_df.head(10), 
                    x='tingkat_kepercayaan', 
                    y='item_rekomendasi',
                    orientation='h',
                    color='lift',
                    title="Skor Confidence Cross-Selling"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Tidak ada rekomendasi cross-selling yang ditemukan dengan ambang batas saat ini.")
        
        with col2:
            st.subheader("📈 Rekomendasi Upselling")
            if up_recs:
                for i, rec in enumerate(up_recs[:5]):
                    with st.expander(f"Paket {i+1} (Confidence: {rec['tingkat_kepercayaan']:.3f})"):
                        st.write(f"**Item Rekomendasi:** {', '.join(rec['item_rekomendasi'])}")
                        st.write(f"**Ukuran Paket:** {rec['ukuran_paket']} item")
                        st.write(f"**Support:** {rec['dukungan']:.3f}")
                        st.write(f"**Lift:** {rec['lift']:.3f}")
            else:
                st.info("Tidak ada rekomendasi upselling yang ditemukan dengan ambang batas saat ini.")

def show_basket_recommendations(model):
    """Menampilkan rekomendasi berdasarkan keranjang saat ini"""
    st.header("🛍️ Rekomendasi Keranjang Belanja")
    
    # Multi-select untuk item keranjang
    available_items = model.all_items if hasattr(model, 'all_items') else []
    
    if not available_items:
        st.error("Tidak ada item tersedia dalam model.")
        return
    
    # Pemilihan keranjang
    st.subheader("🛒 Buat Keranjang Belanja Anda")
    selected_basket = st.multiselect(
        "Tambahkan item ke keranjang Anda:",
        available_items,
        help="Pilih beberapa item yang saat ini ada dalam keranjang belanja Anda"
    )
    
    if not selected_basket:
        st.info("Silakan tambahkan beberapa item ke keranjang Anda untuk mendapatkan rekomendasi.")
        return
    
    # Konfigurasi
    col1, col2 = st.columns(2)
    with col1:
        confidence_threshold = st.slider(
            "Batas Minimum Confidence", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.2, 
            step=0.05,
            key="basket_confidence"
        )
    
    with col2:
        top_n = st.slider(
            "Jumlah Rekomendasi", 
            min_value=1, 
            max_value=20, 
            value=10, 
            step=1,
            key="basket_top_n"
        )
    
    # Tampilan keranjang saat ini
    st.subheader("📋 Keranjang Saat Ini")
    basket_df = pd.DataFrame({'Item dalam Keranjang': selected_basket})
    st.dataframe(basket_df, use_container_width=True)
    
    if st.button("🎯 Dapatkan Rekomendasi Keranjang"):
        # Dapatkan rekomendasi
        recommendations = model.get_basket_recommendations(
            selected_basket, 
            top_n=top_n, 
            min_confidence=confidence_threshold
        )
        
        if recommendations:
            st.subheader("✨ Rekomendasi Item Tambahan")
            
            # Buat DataFrame untuk tampilan yang lebih baik
            rec_data = []
            for rec in recommendations:
                rec_data.append({
                    'Item Rekomendasi': rec['item_rekomendasi'],
                    'Rata-rata Confidence': round(rec['rata_kepercayaan'], 3),
                    'Rata-rata Support': round(rec['rata_dukungan'], 3),
                    'Rata-rata Lift': round(rec['rata_lift'], 3),
                    'Aturan Pendukung': rec['aturan_pendukung'],
                    'Berdasarkan Item': ', '.join(rec['item_pendukung'][:3]) + ('...' if len(rec['item_pendukung']) > 3 else '')
                })
            
            rec_df = pd.DataFrame(rec_data)
            st.dataframe(rec_df, use_container_width=True)
            
            # Visualisasi
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Skor Confidence', 'Nilai Lift'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Grafik confidence
            fig.add_trace(
                go.Bar(
                    name='Confidence',
                    x=rec_df['Item Rekomendasi'][:10],
                    y=rec_df['Rata-rata Confidence'][:10],
                    marker_color='lightblue'
                ),
                row=1, col=1
            )
            
            # Grafik lift
            fig.add_trace(
                go.Bar(
                    name='Lift',
                    x=rec_df['Item Rekomendasi'][:10],
                    y=rec_df['Rata-rata Lift'][:10],
                    marker_color='lightgreen'
                ),
                row=1, col=2
            )
            
            fig.update_xaxes(tickangle=45)
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("Tidak ada rekomendasi yang ditemukan untuk keranjang saat ini dengan ambang batas confidence yang ditentukan.")

def show_advanced_analytics(model):
    """Menampilkan analitik lanjutan dan wawasan"""
    st.header("📊 Analitik Lanjutan")
    
    # Analisis aturan
    if hasattr(model, 'association_rules'):
        st.subheader("📈 Analisis Aturan Asosiasi")
        
        rules_data = []
        for i in range(len(model.association_rules['antecedents'])):
            rules_data.append({
                'Anteseden': ', '.join(model.association_rules['antecedents'][i]),
                'Konsekuen': ', '.join(model.association_rules['consequents'][i]),
                'Support': round(model.association_rules['support'][i], 3),
                'Confidence': round(model.association_rules['confidence'][i], 3),
                'Lift': round(model.association_rules['lift'][i], 3)
            })
        
        rules_df = pd.DataFrame(rules_data)
        
        # Filter
        col1, col2, col3 = st.columns(3)
        with col1:
            min_support = st.slider("Min Support", 0.0, 1.0, 0.01, 0.01)
        with col2:
            min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.2, 0.01)
        with col3:
            min_lift = st.slider("Min Lift", 0.0, 10.0, 1.0, 0.1)
        
        # Filter aturan
        filtered_rules = rules_df[
            (rules_df['Support'] >= min_support) &
            (rules_df['Confidence'] >= min_confidence) &
            (rules_df['Lift'] >= min_lift)
        ]
        
        st.write(f"Menampilkan {len(filtered_rules)} aturan dari {len(rules_df)} total aturan")
        
        # Tampilkan aturan yang difilter
        if len(filtered_rules) > 0:
            st.dataframe(filtered_rules.head(20), use_container_width=True)
            
            # Scatter plot
            fig = px.scatter(
                filtered_rules,
                x='Support',
                y='Confidence',
                color='Lift',
                size='Lift',
                hover_data=['Anteseden', 'Konsekuen'],
                title="Aturan Asosiasi: Support vs Confidence (diwarnai berdasarkan Lift)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot distribusi
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(
                    filtered_rules,
                    x='Confidence',
                    nbins=20,  # Ganti 'bins' dengan 'nbins'
                    title="Distribusi Nilai Confidence"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                fig_hist2 = px.histogram(
                    filtered_rules,
                    x='Lift',
                    nbins=20,  # Ganti 'bins' dengan 'nbins'
                    title="Distribusi Nilai Lift"
                )
                st.plotly_chart(fig_hist2, use_container_width=True)
        else:
            st.info("Tidak ada aturan yang sesuai dengan kriteria filter saat ini.")
    
    # Analisis frekuensi item
    if hasattr(model, 'item_names') and hasattr(model, 'item_frequencies'):
        st.subheader("📊 Analisis Frekuensi Item")
        
        freq_df = pd.DataFrame({
            'Item': model.item_names,
            'Frekuensi': model.item_frequencies
        })
        
        # Pemilih Top N item
        top_n_items = st.slider("Tampilkan Top N Item", 5, 50, 20)
        top_items_df = freq_df.head(top_n_items)
        
        # Grafik bar
        fig = px.bar(
            top_items_df,
            x='Frekuensi',
            y='Item',
            orientation='h',
            title=f"Top {top_n_items} Item Paling Sering Muncul"
        )
        fig.update_layout(height=max(400, top_n_items * 20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribusi frekuensi
        fig_dist = px.histogram(
            freq_df,
            x='Frekuensi',
            nbins=30,
            title="Distribusi Frekuensi Item"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

if __name__ == "__main__":
    main()