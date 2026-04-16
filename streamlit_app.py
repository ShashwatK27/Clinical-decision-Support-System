"""
Streamlit Web Application for Clinical Decision Support System
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.parser import parse_prescription
from preprocessing.cleaner import clean_medications, build_embedding_text
from mapping.fuzzy_match import correct_drug_list
from mapping.condition_mapper import ConditionMapper
from embeddings.embedding import get_embedding
from vector_db.store import VectorStore
from utils.logger_config import get_logger

logger = get_logger("streamlit_app")

# Page configuration
st.set_page_config(
    page_title="🩺 Clinical Decision Support System",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
        padding: 10px 20px;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #28a745;
        border-radius: 5px;
        padding: 10px;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #17a2b8;
        border-radius: 5px;
        padding: 10px;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state (LAZY - deferred initialization)
if "store" not in st.session_state:
    st.session_state.store = None
    st.session_state.mapper = None
    st.session_state.system_ready = False

@st.cache_resource
def init_system():
    """Initialize or load vector store and condition mapper (deferred, cached)"""
    try:
        mapper = ConditionMapper()
        
        # Try to load the pre-built store from build_db.py
        store = VectorStore.load("vector_store.pkl")
        
        if store is None:
            # Fallback: Create empty store with just example cases
            logger.warning("No pre-built vector store found. Creating new store with example cases.")
            store = VectorStore()
            
            example_cases = [
                (["ibuprofen"], ["pain", "inflammation"]),
                (["metformin"], ["diabetes"]),
                (["paracetamol"], ["fever"]),
            ]
            
            for drugs, conditions in example_cases:
                text = " ".join(drugs + conditions)
                vector = get_embedding(text)
                metadata = {"drugs": drugs, "conditions": conditions, "original_text": text}
                store.add(vector, metadata)
            
            logger.info(f"Created fallback store with {len(store.vectors)} example cases")
        else:
            logger.info(f"Loaded pre-built vector store with {len(store.vectors)} cases")
        
        return store, mapper
    except Exception as e:
        logger.error(f"System initialization error: {e}")
        return None, None

# Try to initialize system once
if st.session_state.store is None and st.session_state.mapper is None:
    st.session_state.store, st.session_state.mapper = init_system()

# Header
st.title("🩺 Clinical Decision Support System")
st.markdown("*Intelligent prescribing guidance powered by AI*")

# Check system status
system_ready = st.session_state.store is not None and st.session_state.mapper is not None

if not system_ready:
    st.warning("⚠️ **System initializing...** The embedding model is downloading. Please wait and refresh the page (F5) in 30-60 seconds.")
    st.stop()

st.success("✅ System ready!")

# Sidebar
with st.sidebar:
    st.header("⚙️ System Status")
    st.metric("Vector DB Size", len(st.session_state.store.vectors) if st.session_state.store else 0)
    st.metric("Known Drugs", len(st.session_state.mapper.knowledge_base) if st.session_state.mapper else 0)
    
    st.divider()
    
    st.header("📚 About")
    st.markdown("""
    This system helps healthcare providers by:
    - 🔍 Analyzing prescriptions
    - 🧠 Predicting clinical conditions
    - 📊 Finding similar cases
    - 💡 Providing recommendations
    """)
    
    st.divider()
    
    st.header("🔧 Configuration")
    fuzzy_threshold = st.slider("Fuzzy Match Threshold", 0.70, 0.95, 0.85)
    search_threshold = st.slider("Search Threshold", 0.20, 0.80, 0.35)
    top_k = st.slider("Top K Similar Cases", 1, 5, 3)

# Main content - Tabs
tab1, tab2, tab3 = st.tabs(["🩋 Prescription Analysis", "📖 Examples", "📊 System Info"])

# ============================================================================
# TAB 1: PRESCRIPTION ANALYSIS
# ============================================================================
with tab1:
    st.header("Enter Prescription")
    
    # Input options
    input_method = st.radio("Input method:", ["Text Input", "Paste Example"])
    
    if input_method == "Paste Example":
        examples = {
            "Example 1: Typo Tolerance": "iboprofen 200mg and metaformin 500mg",
            "Example 2: Single Drug": "paracetamol 650 mg",
            "Example 3: Complex Text": "Patient should take ibuprofen daily for pain",
            "Example 4: Structured": "medications: aspirin 100mg, lisinopril 10mg",
        }
        selected_example = st.selectbox("Choose an example:", list(examples.keys()))
        prescription_text = examples[selected_example]
    else:
        prescription_text = st.text_area(
            "Enter prescription text:",
            placeholder="e.g., ibuprofen 200mg and metformin 500mg",
            height=100
        )
    
    if st.button("🔍 Analyze Prescription", type="primary", use_container_width=True):
        if not prescription_text.strip():
            st.warning("⚠️ Please enter a prescription.")
        else:
            logger.info(f"Analyzing prescription: {prescription_text}")
            
            with st.spinner("🔄 Processing prescription..."):
                # Step 1: Parse
                st.subheader("1️⃣ Parsing")
                parsed = parse_prescription(prescription_text)
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Raw Drugs Extracted:**")
                    st.code(str(parsed['drugs']), language="text")
                with col2:
                    st.write("**Medications:**")
                    st.code(str(parsed['medications']), language="text")
                
                # Step 2: Fuzzy Correction
                st.subheader("2️⃣ Fuzzy Matching & Correction")
                corrected_drugs = correct_drug_list(parsed["drugs"])
                
                if corrected_drugs:
                    st.success(f"✅ Corrected drugs: **{', '.join(corrected_drugs)}**")
                    logger.info(f"Corrected drugs: {corrected_drugs}")
                else:
                    st.warning("⚠️ No recognized drugs found after correction.")
                    logger.warning(f"No recognized drugs found for: {parsed['drugs']}")
                    corrected_drugs = parsed["drugs"]
                
                # Step 3: Semantic Search
                st.subheader("3️⃣ Semantic Search")
                cleaned_meds = clean_medications(parsed["medications"])
                query_text = build_embedding_text(corrected_drugs, cleaned_meds)
                query_vector = get_embedding(query_text)
                search_results = st.session_state.store.search(
                    query_vector, 
                    top_k=top_k, 
                    threshold=search_threshold
                )
                
                if search_results:
                    top_scores = [score for _, score in search_results]
                    if all(score < search_threshold for score in top_scores):
                        st.warning("⚠️ Similarity is low for all returned cases. The results are weak semantic matches.")
                    else:
                        st.success(f"Found **{len(search_results)}** similar case(s)")
                    for i, (meta, score) in enumerate(search_results, 1):
                        with st.expander(f"Case {i} (Similarity: {score:.3f})"):
                            st.write(f"**Drugs:** {', '.join(meta.get('drugs', []))}")
                            st.write(f"**Conditions:** {', '.join(meta.get('conditions', []))}")
                            st.progress(float(score), text=f"Similarity: {score:.1%}")
                    logger.info(f"Found {len(search_results)} similar cases")
                else:
                    st.info("ℹ️ No similar cases found in database.")
                
                # Step 4: Condition Prediction
                st.subheader("4️⃣ Clinical Condition Prediction")
                predictions = st.session_state.mapper.predict(
                    corrected_drugs, 
                    vector_results=search_results
                )
                
                if predictions:
                    st.success(f"✅ Predicted **{len(predictions)}** condition(s)")
                    
                    # Create columns for condition cards
                    cols = st.columns(min(3, len(predictions)))
                    for idx, pred in enumerate(predictions[:3]):
                        with cols[idx % len(cols)]:
                            with st.container():
                                st.metric(
                                    pred['condition_label'].upper(),
                                    f"{pred['confidence']:.2f}",
                                    delta="Confidence"
                                )
                    
                    # Show all predictions in a table
                    st.write("**All Predictions:**")
                    prediction_data = [
                        {
                            "Condition": pred['condition_label'],
                            "Confidence": f"{pred['confidence']:.2f}",
                            "Type": "Rule-based" if pred['confidence'] == 2.0 else "Vector-enhanced"
                        }
                        for pred in predictions
                    ]
                    st.dataframe(prediction_data, use_container_width=True)
                    logger.info(f"Predicted conditions: {[p['condition_label'] for p in predictions]}")
                else:
                    st.info("ℹ️ No conditions could be predicted. Try different drug names.")
                
                # Step 5: Clinical Recommendations
                st.subheader("5️⃣ Clinical Recommendations")
                if predictions:
                    recommendations = {
                        "pain": "✓ Consider pain management strategies (NSAIDs, analgesics)",
                        "inflammation": "✓ Monitor for inflammatory signs; consider anti-inflammatory therapy",
                        "diabetes": "✓ Monitor blood glucose levels; verify dosing",
                        "fever": "✓ Monitor vital signs; consider antipyretics if indicated",
                        "hypertension": "✓ Monitor BP; check for drug interactions",
                        "high cholesterol": "✓ Verify statin compliance; monitor lipid levels",
                        "bacterial infection": "✓ Verify appropriate antibiotic; check sensitivity",
                        "neuropathy": "✓ Monitor for symptom improvement; assess tolerability",
                        "seizures": "✓ Monitor seizure frequency; check drug levels",
                        "thyroid disorder": "✓ Monitor TSH levels; verify compliance",
                        "acid reflux": "✓ Verify PPI dosing; check for long-term safety",
                        "gerd": "✓ Lifestyle modifications + pharmacotherapy assessment",
                        "heart disease prevention": "✓ Monitor for adverse effects; assess cardiovascular risk",
                        "autoimmune disease": "✓ Monitor inflammatory markers; assess immune function",
                    }
                    
                    for pred in predictions[:5]:
                        cond = pred['condition_label'].lower()
                        rec = recommendations.get(cond, "✓ Monitor patient status regularly")
                        st.markdown(f"**{pred['condition_label']}:** {rec}")
                else:
                    st.info("No recommendations available without predictions.")

# ============================================================================
# TAB 2: EXAMPLES
# ============================================================================
with tab2:
    st.header("📖 Common Examples")
    
    examples = {
        "Pain Management": {
            "prescription": "ibuprofen 200mg twice daily",
            "description": "Common NSAID for pain and inflammation"
        },
        "Diabetes Management": {
            "prescription": "metformin 500mg with meals",
            "description": "First-line oral antidiabetic agent"
        },
        "Fever Reduction": {
            "prescription": "paracetamol 650mg every 4 hours",
            "description": "Acetaminophen for fever and pain"
        },
        "Multi-Drug Scenario": {
            "prescription": "iboprofen 200mg and metaformin 500mg",
            "description": "Demonstrates multi-drug reasoning with typo correction"
        },
        "Complex Prescription": {
            "prescription": "Patient should take ibuprofen daily for pain and inflammation",
            "description": "Unstructured natural language prescription"
        },
    }
    
    selected = st.selectbox("Choose an example:", list(examples.keys()))
    example = examples[selected]
    
    st.write(f"**Prescription:** {example['prescription']}")
    st.write(f"**Purpose:** {example['description']}")
    
    if st.button("📊 Run Example Analysis"):
        # Reuse logic from tab 1
        parsed = parse_prescription(example['prescription'])
        corrected = correct_drug_list(parsed["drugs"])
        predictions = st.session_state.mapper.predict(corrected)
        
        st.write("**Predicted Conditions:**")
        if predictions:
            for pred in predictions:
                st.write(f"- {pred['condition_label']} (confidence: {pred['confidence']})")
        else:
            st.info("No conditions predicted")

# ============================================================================
# TAB 3: SYSTEM INFO
# ============================================================================
with tab3:
    st.header("📊 System Information")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Vector DB Cases", len(st.session_state.store.vectors))
    with col2:
        st.metric("Known Drugs", len(st.session_state.mapper.knowledge_base))
    with col3:
        st.metric("Embedding Model", "MiniLM-L6")
    
    st.divider()
    
    st.subheader("🔧 Embedding Configuration")
    st.write("""
    - **Model**: sentence-transformers/all-MiniLM-L6-v2
    - **Dimension**: 384-dimensional vectors
    - **Normalization**: L2 normalization applied
    - **Distance Metric**: Cosine similarity
    """)
    
    st.subheader("📚 Knowledge Base")
    st.write(f"""
    - **Known Drugs**: {len(st.session_state.mapper.knowledge_base)}
    - **Prediction Method**: Rule-based + Vector-based hybrid
    - **Fuzzy Match**: difflib with 85% threshold
    - **Confidence Filter**: 40% of max score (adaptive)
    """)
    
    st.subheader("🔍 Known Drug Conditions")
    if st.checkbox("Show all drug mappings"):
        kb = st.session_state.mapper.knowledge_base
        for drug, conditions in sorted(kb.items())[:20]:
            st.write(f"**{drug}**: {', '.join(conditions)}")
        if len(kb) > 20:
            st.info(f"... and {len(kb) - 20} more drugs")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    🩺 Clinical Decision Support System v1.1 | Powered by Streamlit
    </div>
""", unsafe_allow_html=True)

logger.info("Streamlit app session completed successfully")
