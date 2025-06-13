import streamlit as st
import h5py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

class H5MarketBasketRecommender:
    """
    Market Basket Recommender yang menggunakan model H5
    Kompatibel dengan arsitektur TensorFlow/Keras untuk ML pipeline
    """
    
    def __init__(self, h5_model_path):
        """Load model dari file H5"""
        self.model_path = h5_model_path
        self.load_model()
    
    def load_model(self):
        """Load semua komponen model dari H5 file"""
        with h5py.File(self.model_path, 'r') as f:
            
            # Load metadata
            self.metadata = dict(f['model_metadata'].attrs)
            
            # Load frequent itemsets
            if 'frequent_itemsets' in f:
                freq_group = f['frequent_itemsets']
                self.frequent_itemsets = {
                    'itemsets': [eval(s.decode('utf-8')) for s in freq_group['itemsets'][:]],
                    'support': freq_group['support'][:],
                    'length': freq_group['length'][:]
                }
            
            # Load association rules
            if 'association_rules' in f:
                rules_group = f['association_rules']
                self.association_rules = {
                    'antecedents': [eval(s.decode('utf-8')) for s in rules_group['antecedents'][:]],
                    'consequents': [eval(s.decode('utf-8')) for s in rules_group['consequents'][:]],
                    'support': rules_group['support'][:],
                    'confidence': rules_group['confidence'][:],
                    'lift': rules_group['lift'][:]
                }
            
            # Load cross-selling rules
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
            
            # Load upselling rules
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
            
            # Load item statistics
            if 'item_statistics' in f:
                items_group = f['item_statistics']
                self.item_names = [s.decode('utf-8') for s in items_group['item_names'][:]]
                self.item_frequencies = items_group['item_frequencies'][:]
                self.all_items = [s.decode('utf-8') for s in items_group['all_items'][:]]
            
            # Load performance metrics
            if 'performance_metrics' in f:
                self.performance_metrics = dict(f['performance_metrics'].attrs)
    
    def get_cross_selling_recommendations(self, item, top_n=5, min_confidence=0.3):
        """Dapatkan rekomendasi cross-selling untuk item tertentu"""
        if not self.cross_selling_rules:
            return []
        
        recommendations = []
        
        for i, antecedent in enumerate(self.cross_selling_rules['antecedents']):
            if item in antecedent and self.cross_selling_rules['confidence'][i] >= min_confidence:
                consequent = self.cross_selling_rules['consequents'][i]
                if len(consequent) == 1:  # Cross-selling: 1 -> 1
                    recommendations.append({
                        'recommended_item': consequent[0],
                        'confidence': float(self.cross_selling_rules['confidence'][i]),
                        'support': float(self.cross_selling_rules['support'][i]),
                        'lift': float(self.cross_selling_rules['lift'][i]),
                        'rule_strength': 'cross_selling'
                    })
        
        # Sort by confidence and return top N
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        return recommendations[:top_n]
    
    def get_upselling_recommendations(self, item, top_n=5, min_confidence=0.25):
        """Dapatkan rekomendasi upselling untuk item tertentu"""
        if not self.upselling_rules:
            return []
        
        recommendations = []
        
        for i, antecedent in enumerate(self.upselling_rules['antecedents']):
            if item in antecedent and self.upselling_rules['confidence'][i] >= min_confidence:
                consequents = self.upselling_rules['consequents'][i]
                recommendations.append({
                    'recommended_items': consequents,
                    'confidence': float(self.upselling_rules['confidence'][i]),
                    'support': float(self.upselling_rules['support'][i]),
                    'lift': float(self.upselling_rules['lift'][i]),
                    'rule_strength': 'upselling',
                    'bundle_size': len(consequents)
                })
        
        # Sort by confidence and return top N
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        return recommendations[:top_n]
    
    def get_basket_recommendations(self, basket_items, top_n=10, min_confidence=0.2):
        """Rekomendasi berdasarkan keranjang belanja saat ini"""
        all_recommendations = {}
        
        for item in basket_items:
            if item in self.all_items:
                # Get cross-selling recommendations
                cross_recs = self.get_cross_selling_recommendations(item, top_n=20, min_confidence=min_confidence)
                
                for rec in cross_recs:
                    rec_item = rec['recommended_item']
                    if rec_item not in basket_items:  # Jangan rekomendasikan item yang sudah ada
                        if rec_item not in all_recommendations:
                            all_recommendations[rec_item] = {
                                'total_confidence': 0,
                                'total_support': 0,
                                'total_lift': 0,
                                'rule_count': 0,
                                'supporting_items': []
                            }
                        
                        all_recommendations[rec_item]['total_confidence'] += rec['confidence']
                        all_recommendations[rec_item]['total_support'] += rec['support']
                        all_recommendations[rec_item]['total_lift'] += rec['lift']
                        all_recommendations[rec_item]['rule_count'] += 1
                        all_recommendations[rec_item]['supporting_items'].append(item)
        
        # Calculate averages and create final recommendations
        final_recommendations = []
        for item, stats in all_recommendations.items():
            if stats['rule_count'] > 0:
                final_recommendations.append({
                    'recommended_item': item,
                    'avg_confidence': stats['total_confidence'] / stats['rule_count'],
                    'avg_support': stats['total_support'] / stats['rule_count'],
                    'avg_lift': stats['total_lift'] / stats['rule_count'],
                    'supporting_rules': stats['rule_count'],
                    'supporting_items': stats['supporting_items']
                })
        
        # Sort by average confidence
        final_recommendations.sort(key=lambda x: x['avg_confidence'], reverse=True)
        return final_recommendations[:top_n]
    
    def get_model_info(self):
        """Informasi model dan performa"""
        return {
            'model_metadata': self.metadata,
            'performance_metrics': self.performance_metrics if hasattr(self, 'performance_metrics') else {},
            'total_items': len(self.all_items) if hasattr(self, 'all_items') else 0,
            'top_items': dict(zip(self.item_names[:10], self.item_frequencies[:10])) if hasattr(self, 'item_names') else {}
        }

@st.cache_resource
def load_model(model_path):
    """Cache the model loading to improve performance"""
    try:
        return H5MarketBasketRecommender(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    # Title and Header
    st.title("🛒 Market Basket Analysis Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    
    # Model path input
    model_path = st.sidebar.text_input(
        "Model H5 Path", 
        value="market_basket_model.h5",
        help="Enter the path to your H5 model file"
    )
    
    # Load model
    if st.sidebar.button("🔄 Load Model"):
        st.session_state.model = load_model(model_path)
        if st.session_state.model:
            st.sidebar.success("✅ Model loaded successfully!")
        else:
            st.sidebar.error("❌ Failed to load model")
    
    # Initialize model if not exists
    if 'model' not in st.session_state:
        st.session_state.model = load_model(model_path)
    
    if st.session_state.model is None:
        st.error("⚠️ Please ensure the model file exists and try loading again.")
        return
    
    model = st.session_state.model
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["📈 Model Overview", "🎯 Single Item Recommendations", "🛍️ Basket Recommendations", "📊 Advanced Analytics"]
    )
    
    if page == "📈 Model Overview":
        show_model_overview(model)
    elif page == "🎯 Single Item Recommendations":
        show_single_item_recommendations(model)
    elif page == "🛍️ Basket Recommendations":
        show_basket_recommendations(model)
    elif page == "📊 Advanced Analytics":
        show_advanced_analytics(model)

def show_model_overview(model):
    """Display model overview and statistics"""
    st.header("📈 Model Overview")
    
    # Get model info
    model_info = model.get_model_info()
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Items", 
            model_info['total_items'],
            help="Number of unique items in the dataset"
        )
    
    with col2:
        st.metric(
            "Total Rules", 
            model_info['model_metadata']['total_association_rules'],
            help="Number of association rules generated"
        )
    
    with col3:
        st.metric(
            "Total Transactions", 
            model_info['model_metadata']['total_transactions'],
            help="Number of transactions analyzed"
        )
    
    with col4:
        avg_conf = model_info['performance_metrics'].get('avg_confidence', 0)
        st.metric(
            "Avg Confidence", 
            f"{avg_conf:.3f}",
            help="Average confidence of association rules"
        )
    
    # Model details
    st.subheader("🔧 Model Configuration")
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.info(f"**Algorithm:** {model_info['model_metadata']['algorithm']}")
        st.info(f"**Min Support:** {model_info['model_metadata']['min_support_threshold']}")
        st.info(f"**Min Confidence:** {model_info['model_metadata']['min_confidence_threshold']}")
    
    with config_col2:
        st.info(f"**Created:** {model_info['model_metadata']['created_at'][:19]}")
        st.info(f"**Framework:** {model_info['model_metadata']['framework_version']}")
        st.info(f"**Frequent Itemsets:** {model_info['model_metadata']['total_frequent_itemsets']}")
    
    # Top items chart
    st.subheader("🏆 Top 10 Most Popular Items")
    if model_info['top_items']:
        items_df = pd.DataFrame(
            list(model_info['top_items'].items()),
            columns=['Item', 'Frequency']
        )
        
        fig = px.bar(
            items_df, 
            x='Frequency', 
            y='Item', 
            orientation='h',
            color='Frequency',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    if 'performance_metrics' in model_info and model_info['performance_metrics']:
        st.subheader("📊 Performance Metrics")
        perf_metrics = model_info['performance_metrics']
        
        metrics_df = pd.DataFrame({
            'Metric': ['Support', 'Confidence', 'Lift'],
            'Average': [
                perf_metrics.get('avg_support', 0),
                perf_metrics.get('avg_confidence', 0),
                perf_metrics.get('avg_lift', 0)
            ],
            'Maximum': [
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
        fig.add_trace(go.Bar(name='Average', x=metrics_df['Metric'], y=metrics_df['Average']))
        fig.add_trace(go.Bar(name='Maximum', x=metrics_df['Metric'], y=metrics_df['Maximum']))
        fig.add_trace(go.Bar(name='Minimum', x=metrics_df['Metric'], y=metrics_df['Minimum']))
        
        fig.update_layout(
            title="Association Rules Performance Metrics",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def show_single_item_recommendations(model):
    """Show recommendations for a single item"""
    st.header("🎯 Single Item Recommendations")
    
    # Item selection
    available_items = model.all_items if hasattr(model, 'all_items') else []
    
    if not available_items:
        st.error("No items available in the model.")
        return
    
    selected_item = st.selectbox(
        "Select an item to get recommendations:",
        available_items,
        help="Choose an item to see what other items are frequently bought together"
    )
    
    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        confidence_threshold = st.slider(
            "Minimum Confidence", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.3, 
            step=0.05,
            help="Higher values give more reliable recommendations"
        )
    
    with col2:
        top_n = st.slider(
            "Number of Recommendations", 
            min_value=1, 
            max_value=20, 
            value=10, 
            step=1
        )
    
    if st.button("🔍 Get Recommendations"):
        # Get cross-selling recommendations
        cross_recs = model.get_cross_selling_recommendations(
            selected_item, 
            top_n=top_n, 
            min_confidence=confidence_threshold
        )
        
        # Get upselling recommendations
        up_recs = model.get_upselling_recommendations(
            selected_item, 
            top_n=top_n, 
            min_confidence=confidence_threshold-0.05
        )
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔄 Cross-Selling Recommendations")
            if cross_recs:
                cross_df = pd.DataFrame(cross_recs)
                cross_df['confidence'] = cross_df['confidence'].round(3)
                cross_df['support'] = cross_df['support'].round(3)
                cross_df['lift'] = cross_df['lift'].round(3)
                
                st.dataframe(cross_df, use_container_width=True)
                
                # Visualization
                fig = px.bar(
                    cross_df.head(10), 
                    x='confidence', 
                    y='recommended_item',
                    orientation='h',
                    color='lift',
                    title="Cross-Selling Confidence Scores"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No cross-selling recommendations found with the current threshold.")
        
        with col2:
            st.subheader("📈 Upselling Recommendations")
            if up_recs:
                for i, rec in enumerate(up_recs[:5]):
                    with st.expander(f"Bundle {i+1} (Confidence: {rec['confidence']:.3f})"):
                        st.write(f"**Recommended Items:** {', '.join(rec['recommended_items'])}")
                        st.write(f"**Bundle Size:** {rec['bundle_size']} items")
                        st.write(f"**Support:** {rec['support']:.3f}")
                        st.write(f"**Lift:** {rec['lift']:.3f}")
            else:
                st.info("No upselling recommendations found with the current threshold.")

def show_basket_recommendations(model):
    """Show recommendations based on current basket"""
    st.header("🛍️ Shopping Basket Recommendations")
    
    # Multi-select for basket items
    available_items = model.all_items if hasattr(model, 'all_items') else []
    
    if not available_items:
        st.error("No items available in the model.")
        return
    
    # Basket selection
    st.subheader("🛒 Build Your Shopping Basket")
    selected_basket = st.multiselect(
        "Add items to your basket:",
        available_items,
        help="Select multiple items that are currently in your shopping basket"
    )
    
    if not selected_basket:
        st.info("Please add some items to your basket to get recommendations.")
        return
    
    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        confidence_threshold = st.slider(
            "Minimum Confidence", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.2, 
            step=0.05,
            key="basket_confidence"
        )
    
    with col2:
        top_n = st.slider(
            "Number of Recommendations", 
            min_value=1, 
            max_value=20, 
            value=10, 
            step=1,
            key="basket_top_n"
        )
    
    # Current basket display
    st.subheader("📋 Current Basket")
    basket_df = pd.DataFrame({'Items in Basket': selected_basket})
    st.dataframe(basket_df, use_container_width=True)
    
    if st.button("🎯 Get Basket Recommendations"):
        # Get recommendations
        recommendations = model.get_basket_recommendations(
            selected_basket, 
            top_n=top_n, 
            min_confidence=confidence_threshold
        )
        
        if recommendations:
            st.subheader("✨ Recommended Additional Items")
            
            # Create DataFrame for better display
            rec_data = []
            for rec in recommendations:
                rec_data.append({
                    'Recommended Item': rec['recommended_item'],
                    'Avg Confidence': round(rec['avg_confidence'], 3),
                    'Avg Support': round(rec['avg_support'], 3),
                    'Avg Lift': round(rec['avg_lift'], 3),
                    'Supporting Rules': rec['supporting_rules'],
                    'Based on Items': ', '.join(rec['supporting_items'][:3]) + ('...' if len(rec['supporting_items']) > 3 else '')
                })
            
            rec_df = pd.DataFrame(rec_data)
            st.dataframe(rec_df, use_container_width=True)
            
            # Visualization
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Confidence Scores', 'Lift Values'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Confidence chart
            fig.add_trace(
                go.Bar(
                    name='Confidence',
                    x=rec_df['Recommended Item'][:10],
                    y=rec_df['Avg Confidence'][:10],
                    marker_color='lightblue'
                ),
                row=1, col=1
            )
            
            # Lift chart
            fig.add_trace(
                go.Bar(
                    name='Lift',
                    x=rec_df['Recommended Item'][:10],
                    y=rec_df['Avg Lift'][:10],
                    marker_color='lightgreen'
                ),
                row=1, col=2
            )
            
            fig.update_xaxes(tickangle=45)
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("No recommendations found for the current basket with the specified confidence threshold.")

def show_advanced_analytics(model):
    """Show advanced analytics and insights"""
    st.header("📊 Advanced Analytics")
    
    # Rules analysis
    if hasattr(model, 'association_rules'):
        st.subheader("📈 Association Rules Analysis")
        
        rules_data = []
        for i in range(len(model.association_rules['antecedents'])):
            rules_data.append({
                'Antecedents': ', '.join(model.association_rules['antecedents'][i]),
                'Consequents': ', '.join(model.association_rules['consequents'][i]),
                'Support': round(model.association_rules['support'][i], 3),
                'Confidence': round(model.association_rules['confidence'][i], 3),
                'Lift': round(model.association_rules['lift'][i], 3)
            })
        
        rules_df = pd.DataFrame(rules_data)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            min_support = st.slider("Min Support", 0.0, 1.0, 0.01, 0.01)
        with col2:
            min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.2, 0.01)
        with col3:
            min_lift = st.slider("Min Lift", 0.0, 10.0, 1.0, 0.1)
        
        # Filter rules
        filtered_rules = rules_df[
            (rules_df['Support'] >= min_support) &
            (rules_df['Confidence'] >= min_confidence) &
            (rules_df['Lift'] >= min_lift)
        ]
        
        st.write(f"Showing {len(filtered_rules)} rules out of {len(rules_df)} total rules")
        
        # Display filtered rules
        if len(filtered_rules) > 0:
            st.dataframe(filtered_rules.head(20), use_container_width=True)
            
            # Scatter plot
            fig = px.scatter(
                filtered_rules,
                x='Support',
                y='Confidence',
                color='Lift',
                size='Lift',
                hover_data=['Antecedents', 'Consequents'],
                title="Association Rules: Support vs Confidence (colored by Lift)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution plots
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(
                    filtered_rules,
                    x='Confidence',
                    bins=20,
                    title="Distribution of Confidence Values"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                fig_hist2 = px.histogram(
                    filtered_rules,
                    x='Lift',
                    bins=20,
                    title="Distribution of Lift Values"
                )
                st.plotly_chart(fig_hist2, use_container_width=True)
        else:
            st.info("No rules match the current filter criteria.")
    
    # Item frequency analysis
    if hasattr(model, 'item_names') and hasattr(model, 'item_frequencies'):
        st.subheader("📊 Item Frequency Analysis")
        
        freq_df = pd.DataFrame({
            'Item': model.item_names,
            'Frequency': model.item_frequencies
        })
        
        # Top N items selector
        top_n_items = st.slider("Show Top N Items", 5, 50, 20)
        top_items_df = freq_df.head(top_n_items)
        
        # Bar chart
        fig = px.bar(
            top_items_df,
            x='Frequency',
            y='Item',
            orientation='h',
            title=f"Top {top_n_items} Most Frequent Items"
        )
        fig.update_layout(height=max(400, top_n_items * 20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Frequency distribution
        fig_dist = px.histogram(
            freq_df,
            x='Frequency',
            bins=30,
            title="Distribution of Item Frequencies"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

if __name__ == "__main__":
    main()