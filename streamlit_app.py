"""
Streamlit Web Application for Clinical Decision Support System
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.parser import parse_prescription
from preprocessing.cleaner import clean_medications, build_embedding_text
from mapping.fuzzy_match import correct_drug_list
from mapping.condition_mapper import ConditionMapper
from mapping.drug_interactions import check_interactions
from mapping.dosage_validator import extract_dosages, validate_dosages
from utils.pdf_report import generate_pdf_report
from utils.rxnorm_api import validate_drug_list
from embeddings.embedding import get_embedding
from vector_db.store import VectorStore
from utils.logger_config import get_logger
from utils.helpers import sanitize_log_text

logger = get_logger("streamlit_app")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🩺 Clinical Decision Support System",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTabs [data-baseweb="tab-list"] button { font-size: 16px; padding: 10px 20px; }
    .severe-box  { background:#fde8e8; border:1.5px solid #e53e3e; border-radius:6px; padding:10px; margin:6px 0; }
    .moderate-box{ background:#fefcbf; border:1.5px solid #d69e2e; border-radius:6px; padding:10px; margin:6px 0; }
    .mild-box    { background:#e6fffa; border:1.5px solid #38a169; border-radius:6px; padding:10px; margin:6px 0; }
    .history-card{ background:#fff; border:1px solid #dee2e6; border-radius:8px; padding:14px; margin:8px 0; }
    </style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for key in ("store", "mapper", "analysis_history", "last_drugs"):
    if key not in st.session_state:
        st.session_state[key] = None if key != "analysis_history" else []

# ── Noise filter (shared between Tab 1 and history display) ───────────────────
_NOISE_KW = [
    "inhibitor", "agonist", "antagonist", "blocker", "receptor",
    "enzyme", "cytochrome", "transporter", "substrate", "inducer",
    "reuptake", "channel", "modulator", "pump", "synthase",
    "kinase", "allergenic extract", "food allergen", "plant allergen",
    "standardized", "non-standardized", "pharmacological",
]

def _clean_conditions(raw: list) -> list:
    return [c for c in raw if not any(kw in c.lower() for kw in _NOISE_KW)]

# ── System init ────────────────────────────────────────────────────────────────
@st.cache_resource
def init_system():
    try:
        mapper = ConditionMapper()
        store = VectorStore.load("vector_store")

        if store is None:
            logger.warning("No pre-built vector store found — using 3-case fallback.")
            store = VectorStore()
            for drugs, conditions in [
                (["ibuprofen"], ["pain", "inflammation"]),
                (["metformin"], ["diabetes"]),
                (["paracetamol"], ["fever"]),
            ]:
                text = " ".join(drugs + conditions)
                store.add(get_embedding(text), {"drugs": drugs, "conditions": conditions, "original_text": text})
        else:
            logger.info(f"Loaded pre-built vector store with {len(store.vectors)} cases")

        return store, mapper
    except Exception as e:
        logger.error(f"System initialization error: {e}")
        return None, None

if st.session_state.store is None:
    st.session_state.store, st.session_state.mapper = init_system()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🩺 Clinical Decision Support System")
st.markdown("*Intelligent prescribing guidance powered by AI*")

if st.session_state.store is None:
    st.warning("⚠️ **System initializing...** Please wait and refresh (F5) in 30–60 seconds.")
    st.stop()

st.success("✅ System ready!")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Status")
    st.metric("Vector DB Cases", len(st.session_state.store.vectors))
    st.metric("Known Drugs", len(st.session_state.mapper.knowledge_base))
    st.metric("Analyses This Session", len(st.session_state.analysis_history))
    st.divider()
    st.header("📚 About")
    st.markdown("""
    This system helps healthcare providers by:
    - 🔍 Analysing prescriptions
    - ⚠️ Detecting drug interactions
    - 🧠 Predicting clinical conditions
    - 📊 Finding similar historical cases
    - 💡 Providing clinical recommendations
    """)
    st.divider()
    st.header("🔧 Configuration")
    fuzzy_threshold = st.slider("Fuzzy Match Threshold", 0.70, 0.95, 0.88)
    search_threshold = st.slider("Search Threshold", 0.20, 0.80, 0.35)
    top_k = st.slider("Top K Similar Cases", 1, 5, 3)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🩺 Prescription Analysis",
    "📖 Examples",
    "🕑 History",
    "📊 System Info",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: PRESCRIPTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Enter Prescription")

    input_method = st.radio("Input method:", ["Text Input", "Paste Example"])

    if input_method == "Paste Example":
        _EXAMPLES = {
            "Typo tolerance": "iboprofen 200mg and metaformin 500mg",
            "Single drug": "paracetamol 650mg",
            "Complex free-text": "52y female presenting with acid reflux and depression. Prescribed omeprazole 20mg once daily and escitalopram 10mg in the morning.",
            "Multi-drug comorbidity": "Patient prescribed lisinopril 10mg, atorvastatin 40mg and metformin 500mg twice daily",
            "Interaction scenario": "warfarin 5mg once daily and ibuprofen 400mg twice daily",
        }
        selected_example = st.selectbox("Choose an example:", list(_EXAMPLES.keys()))
        prescription_text = _EXAMPLES[selected_example]
        st.text_area("Preview:", prescription_text, height=80, disabled=True)
    else:
        prescription_text = st.text_area(
            "Enter prescription text:",
            placeholder="e.g., ibuprofen 200mg and metformin 500mg",
            height=100,
        )

    if st.button("🔍 Analyze Prescription", type="primary", use_container_width=True):
        if not prescription_text.strip():
            st.warning("⚠️ Please enter a prescription.")
        else:
            logger.info(f"Analyzing prescription: {sanitize_log_text(prescription_text)}")

            # ── Step 1: Parsing ────────────────────────────────────────────
            st.subheader("1️⃣ Parsing")
            parsed = parse_prescription(prescription_text)
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Raw Drugs Extracted:**")
                st.code(str(parsed["drugs"]))
            with col_b:
                st.write("**Medications:**")
                st.code(str(parsed["medications"])[:120] + "…" if len(str(parsed["medications"])) > 120 else str(parsed["medications"]))

            # ── Step 2: Fuzzy Matching ────────────────────────────────────
            st.subheader("2️⃣ Fuzzy Matching & Correction")
            corrected_drugs = correct_drug_list(parsed["drugs"], cutoff=fuzzy_threshold)

            if corrected_drugs:
                st.success(f"✅ Corrected drugs: **{', '.join(corrected_drugs)}**")
                logger.info(f"Corrected drugs: {corrected_drugs}")
            else:
                st.warning("⚠️ No recognised drugs found after correction.")
                corrected_drugs = parsed["drugs"]

            # Persist drugs for the RxNorm section below (survives button reruns)
            st.session_state.last_drugs = corrected_drugs

            # ── Step 3: Dosage Validation ─────────────────────────────────
            st.subheader("3️⃣ Dosage Validation")
            dosages = extract_dosages(prescription_text)
            dose_warnings = validate_dosages(dosages, corrected_drugs)

            if not dosages:
                st.info("ℹ️ No dose values detected (e.g. add '400mg' to your text).")
            elif not dose_warnings:
                detected = ", ".join(
                    f"{d}: {v}{u}" for d, (v, u) in dosages.items()
                    if any(d[:5] == cd[:5] for cd in corrected_drugs)
                )
                st.success(f"✅ All doses within safe limits. {('Detected: ' + detected) if detected else ''}")
            else:
                for dw in dose_warnings:
                    if dw.severity == "high":
                        st.error(
                            f"🔴 **HIGH — {dw.drug.upper()}:** "
                            f"Prescribed **{dw.dose_value} {dw.dose_unit}** exceeds the "
                            f"maximum {dw.limit_type} of **{dw.limit_value} {dw.dose_unit}**.  \n"
                            f"⚠️ {dw.note}"
                        )
                    else:
                        st.warning(
                            f"🟡 **CAUTION — {dw.drug.upper()}:** "
                            f"Prescribed **{dw.dose_value} {dw.dose_unit}** exceeds the "
                            f"recommended {dw.limit_type} of **{dw.limit_value} {dw.dose_unit}**.  \n"
                            f"⚠️ {dw.note}"
                        )

            # ── Step 4: Drug Interactions ─────────────────────────────────
            st.subheader("4️⃣ Drug Interaction Check")
            interactions = check_interactions(corrected_drugs)

            if len(corrected_drugs) < 2:
                st.info("ℹ️ Enter at least 2 drugs to check for interactions.")
            elif not interactions:
                st.success("✅ No known interactions detected between the prescribed drugs.")
            else:
                _severity_icons = {"severe": "🔴", "moderate": "🟡", "mild": "🟢"}
                for ix in interactions:
                    icon = _severity_icons.get(ix.severity, "⚪")
                    css_class = f"{ix.severity}-box"
                    drug_pair = " + ".join(sorted(ix.drugs))
                    st.markdown(
                        f"""<div class="{css_class}">
                        <b>{icon} {ix.severity.upper()}: {drug_pair}</b><br/>
                        <b>Effect:</b> {ix.effect}<br/>
                        <b>Action:</b> {ix.recommendation}
                        </div>""",
                        unsafe_allow_html=True,
                    )

            # ── Step 5: Semantic Search ───────────────────────────────────
            st.subheader("5️⃣ Semantic Search")
            cleaned_meds = clean_medications(parsed["medications"])
            query_vector = get_embedding(build_embedding_text(corrected_drugs, cleaned_meds))
            search_results = st.session_state.store.search(
                query_vector, top_k=top_k, threshold=search_threshold
            )

            if search_results:
                st.success(f"Found **{len(search_results)}** similar case(s)")
                for i, (meta, score) in enumerate(search_results, 1):
                    with st.expander(f"Case {i} (Similarity: {score:.3f})"):
                        drugs_str = ", ".join(meta.get("drugs", []))
                        clean_conds = _clean_conditions(meta.get("conditions", []))
                        conds_str = ", ".join(clean_conds) if clean_conds else "—"
                        st.write(f"**Drugs:** {drugs_str}")
                        st.write(f"**Conditions:** {conds_str}")
                        st.progress(float(score), text=f"Similarity: {score:.1%}")
                logger.info(f"Found {len(search_results)} similar cases")
            else:
                st.info("ℹ️ No similar cases found in database.")

            # ── Step 5: Condition Prediction ──────────────────────────────
            st.subheader("5️⃣ Clinical Condition Prediction")
            predictions = st.session_state.mapper.predict(corrected_drugs, vector_results=search_results)

            if predictions:
                # Normalise confidence to 0–100%
                max_score = max(p["confidence"] for p in predictions)
                for p in predictions:
                    p["confidence_pct"] = round(p["confidence"] / max_score * 100, 1) if max_score else 0.0

                st.success(f"✅ Predicted **{len(predictions)}** condition(s)")
                cols = st.columns(min(3, len(predictions)))
                for idx, pred in enumerate(predictions[:3]):
                    with cols[idx % len(cols)]:
                        st.metric(
                            pred["condition_label"].upper(),
                            f"{pred['confidence_pct']}%",
                            delta="Confidence",
                        )

                # Full table
                st.write("**All Predictions:**")
                st.dataframe(
                    [{"Condition": p["condition_label"], "Confidence": f"{p['confidence_pct']}%"} for p in predictions],
                    use_container_width=True,
                )
                logger.info(f"Predicted conditions: {[p['condition_label'] for p in predictions]}")
            else:
                kb = st.session_state.mapper.knowledge_base
                covered   = [d for d in corrected_drugs if d in kb and kb[d]]
                uncovered = [d for d in corrected_drugs if d not in kb or not kb[d]]

                if uncovered:
                    st.warning(
                        f"⚠️ No conditions predicted. "
                        f"The following drug(s) have no entries in the knowledge base: "
                        f"**{', '.join(uncovered)}**. "
                        f"{'Conditions were found for: ' + ', '.join(covered) + ' but their scores were below threshold.' if covered else 'Consider adding them to the clean seed KB in condition_mapper.py.'}"
                    )
                else:
                    st.info("ℹ️ Drugs were recognised but scores fell below the prediction threshold. Try lowering the Search Threshold in the sidebar.")
                predictions = []

            # ── Step 7: Clinical Recommendations ─────────────────────
            st.subheader("7️⃣ Clinical Recommendations")
            _RECS = {
                "pain":                    "✓ Consider pain management strategies (NSAIDs, analgesics)",
                "inflammation":            "✓ Monitor for inflammatory signs; consider anti-inflammatory therapy",
                "diabetes":                "✓ Monitor blood glucose levels; verify dosing schedule",
                "fever":                   "✓ Monitor vital signs; consider antipyretics if indicated",
                "hypertension":            "✓ Monitor BP regularly; check for drug interactions",
                "high cholesterol":        "✓ Verify statin compliance; monitor lipid panel",
                "bacterial infection":     "✓ Verify appropriate antibiotic choice; check sensitivity",
                "neuropathy":              "✓ Monitor for symptom improvement; assess tolerability",
                "seizures":                "✓ Monitor seizure frequency; check therapeutic drug levels",
                "thyroid disorder":        "✓ Monitor TSH levels; verify compliance and timing",
                "acid reflux":             "✓ Verify PPI dosing; review long-term safety",
                "gerd":                    "✓ Lifestyle modifications + pharmacotherapy assessment",
                "heart disease prevention":"✓ Monitor for adverse effects; assess cardiovascular risk",
                "autoimmune disease":      "✓ Monitor inflammatory markers; assess immune function",
                "rheumatoid arthritis":    "✓ Regular joint assessment; monitor DMARD levels",
                "depression":              "✓ Schedule follow-up in 2–4 weeks; assess response",
                "anxiety":                 "✓ Monitor psychological status; consider CBT referral",
                "insomnia":                "✓ Limit hypnotic use; encourage sleep hygiene",
                "allergic rhinitis":       "✓ Avoid triggers; reassess antihistamine need seasonally",
                "asthma":                  "✓ Check inhaler technique; monitor peak flow",
                "gout":                    "✓ Encourage hydration; dietary uric acid review",
                "angina":                  "✓ Monitor exercise tolerance; review GTN availability",
                "heart failure":           "✓ Daily weight monitoring; fluid restriction counselling",
                "edema":                   "✓ Monitor fluid balance and electrolytes",
                "blood clot prevention":   "✓ Monitor INR regularly; counsel on bleeding signs",
            }
            if predictions:
                for pred in predictions[:6]:
                    cond = pred["condition_label"].lower()
                    rec = _RECS.get(cond, "✓ Monitor patient status regularly")
                    st.markdown(f"**{pred['condition_label']}:** {rec}")
            else:
                st.info("No recommendations available without predictions.")

            # ── Save to history & offer PDF download ───────────────
            # Build recommendation dict for the PDF
            active_recs = {}
            for pred in predictions[:6]:
                cond = pred["condition_label"].lower()
                active_recs[pred["condition_label"]] = _RECS.get(cond, "✓ Monitor patient status regularly")

            st.session_state.analysis_history.append({
                "text": prescription_text[:120],
                "drugs": corrected_drugs,
                "conditions": [p["condition_label"] for p in predictions],
                "interactions": [
                    {"drugs": sorted(ix.drugs), "severity": ix.severity}
                    for ix in interactions
                ],
            })

            # ── PDF Download ─────────────────────────────────────
            st.divider()
            from datetime import datetime as _dt
            report_data = {
                "prescription_text": prescription_text,
                "drugs": corrected_drugs,
                "dosage_warnings": dose_warnings,
                "interactions": interactions,
                "predictions": predictions,
                "recommendations": active_recs,
                "similar_cases": search_results,
                "timestamp": _dt.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            try:
                pdf_bytes = generate_pdf_report(report_data)
                st.download_button(
                    label="📄 Download Clinical Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"cdss_report_{_dt.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as _pdf_err:
                st.warning(f"PDF generation failed: {_pdf_err}")

# ── RxNorm Online Validation (lives OUTSIDE Analyze button so its own button works) ──
_last = st.session_state.get("last_drugs")
if _last:
    with st.expander(f"🌐 RxNorm Online Validation — {', '.join(_last)}", expanded=False):
        _rxkey = f"rxnorm_{','.join(sorted(_last))}"
        if st.button("🔍 Check drugs online via NLM RxNav", key="rxnorm_btn_outer"):
            with st.spinner("Calling NLM RxNav API..."):
                _results = validate_drug_list(_last)
                st.session_state[_rxkey] = _results

        _cached = st.session_state.get(_rxkey)
        if _cached:
            _all_ok = all(r.is_valid for r in _cached)
            if _all_ok:
                st.success("✅ All drugs verified in RxNorm database.")
            else:
                st.warning("⚠️ Some drugs could not be verified — check results below.")
            for r in _cached:
                if r.is_valid:
                    _badge  = "✅"
                    _detail = f"RXCUI: `{r.rxcui}`"
                    if r.canonical_name and r.canonical_name.lower() != r.input_name.lower():
                        _detail += f"  |  RxNorm name: **{r.canonical_name}**"
                    if r.tty:
                        _detail += f"  |  Type: `{r.tty}`"
                else:
                    _badge  = "❓"
                    _detail = f"Not found in RxNorm — {r.error or 'unknown reason'}"
                st.markdown(f"{_badge} **{r.input_name}** — {_detail}")
        else:
            st.caption("Press the button above to validate against the NLM RxNorm database.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: EXAMPLES
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("📖 Common Examples")
    _TAB2_EXAMPLES = {
        "Pain Management":      ("ibuprofen 200mg twice daily", "Common NSAID for pain and inflammation"),
        "Diabetes Management":  ("metformin 500mg twice daily", "First-line oral antidiabetic agent"),
        "Fever Reduction":      ("paracetamol 650mg every 4 hours", "Acetaminophen for fever and mild pain"),
        "Typo Tolerance":       ("iboprofen 200mg and metaformin 500mg", "Tests fuzzy matching with deliberate typos"),
        "Complex Free-Text":    ("52y female with acid reflux and depression. Prescribed omeprazole 20mg and escitalopram 10mg.", "Unstructured clinical note"),
        "Interaction Demo":     ("warfarin 5mg and ibuprofen 400mg", "Demonstrates drug interaction detection"),
    }
    selected = st.selectbox("Choose an example:", list(_TAB2_EXAMPLES.keys()))
    ex_text, ex_desc = _TAB2_EXAMPLES[selected]
    st.write(f"**Prescription:** `{ex_text}`")
    st.write(f"**Purpose:** {ex_desc}")

    if st.button("📊 Run Example Analysis"):
        parsed = parse_prescription(ex_text)
        corrected = correct_drug_list(parsed["drugs"], cutoff=fuzzy_threshold)
        cleaned_meds = clean_medications(parsed["medications"])
        query_vector = get_embedding(build_embedding_text(corrected, cleaned_meds))
        results = st.session_state.store.search(query_vector, top_k=3, threshold=search_threshold)
        predictions = st.session_state.mapper.predict(corrected, vector_results=results)
        interactions = check_interactions(corrected)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Drugs detected:**", ", ".join(corrected) or "none")
            if predictions:
                max_score = max(p["confidence"] for p in predictions)
                st.write("**Predicted conditions:**")
                for p in predictions:
                    pct = round(p["confidence"] / max_score * 100, 1) if max_score else 0
                    st.write(f"  - {p['condition_label']} ({pct}%)")
        with col2:
            if interactions:
                st.write("**⚠️ Interactions:**")
                for ix in interactions:
                    icon = {"severe": "🔴", "moderate": "🟡", "mild": "🟢"}.get(ix.severity, "⚪")
                    st.write(f"  {icon} {' + '.join(sorted(ix.drugs))} — {ix.severity.upper()}")
            else:
                st.write("**✅ No interactions detected**")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🕑 Analysis History")
    history = st.session_state.analysis_history

    if not history:
        st.info("No analyses yet. Run a prescription in the Analysis tab.")
    else:
        if st.button("🗑️ Clear History"):
            st.session_state.analysis_history = []
            st.rerun()

        for i, entry in enumerate(reversed(history), 1):
            with st.expander(f"Analysis #{len(history) - i + 1} — {entry['text'][:60]}…"):
                st.write(f"**Drugs:** {', '.join(entry['drugs']) or 'none'}")
                st.write(f"**Conditions:** {', '.join(entry['conditions']) or 'none predicted'}")
                if entry["interactions"]:
                    _icons = {"severe": "🔴", "moderate": "🟡", "mild": "🟢"}
                    st.write("**Interactions:**")
                    for ix in entry["interactions"]:
                        icon = _icons.get(ix["severity"], "⚪")
                        st.write(f"  {icon} {' + '.join(ix['drugs'])} — {ix['severity'].upper()}")
                else:
                    st.write("**Interactions:** ✅ None detected")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: SYSTEM INFO
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("📊 System Information")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Vector DB Cases", len(st.session_state.store.vectors))
    with col2:
        st.metric("Known Drugs (KB)", len(st.session_state.mapper.knowledge_base))
    with col3:
        st.metric("Embedding Model", "MiniLM-L6")
    with col4:
        st.metric("DDI Rules", "40+")

    st.divider()
    st.subheader("🔧 Embedding Configuration")
    st.write("""
    - **Model**: sentence-transformers/all-MiniLM-L6-v2
    - **Dimension**: 384-dimensional vectors
    - **Normalisation**: L2 normalised
    - **Distance Metric**: Cosine similarity (BLAS dot-product)
    - **Store Format**: NumPy (.npy) + JSON — pickle-free
    """)

    st.subheader("📚 Knowledge Base Coverage")
    st.write(f"""
    - **Clean seed drugs**: 55 (curated clinical labels, always active)
    - **FDA KB**: {len(st.session_state.mapper.knowledge_base)} total entries
    - **Fuzzy Match Cutoff**: {fuzzy_threshold:.2f}
    - **Confidence Display**: Normalised 0–100% relative to top prediction
    """)

    st.subheader("🔍 Sample Drug → Condition Mappings")
    if st.checkbox("Show mappings"):
        kb = st.session_state.mapper.knowledge_base
        sample = {k: v for k, v in list(kb.items())[:25] if v}
        for drug, conditions in sorted(sample.items()):
            st.write(f"**{drug}**: {', '.join(conditions)}")
        if len(kb) > 25:
            st.info(f"… and {len(kb) - 25} more drugs in the knowledge base.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:gray; font-size:12px;'>"
    "🩺 Clinical Decision Support System v2.0 | Powered by Streamlit + sentence-transformers"
    "</div>",
    unsafe_allow_html=True,
)
logger.info("Streamlit app session completed successfully")
