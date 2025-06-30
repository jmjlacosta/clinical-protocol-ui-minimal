"""
Enhanced UI for viewing extraction results with intelligent comparison.

Run with: streamlit run extraction_viewer.py
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import csv

st.set_page_config(
    page_title="Clinical Trial Extraction Results", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Clinical Trial Extraction Results Viewer")
st.markdown("View extraction results with intelligent comparison to ClinicalTrials.gov data")


def load_checkpoint_data() -> Dict[str, Any]:
    """Load all checkpoint files"""
    checkpoints = {}
    checkpoint_dir = Path("checkpoints")
    
    if checkpoint_dir.exists():
        for checkpoint_file in checkpoint_dir.glob("*.json"):
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                key = f"{data['nct_number']}_{data['pdf_type']}"
                checkpoints[key] = data
    
    return checkpoints


def load_ctgov_data(nct_number: str) -> Dict[str, str]:
    """Load CT.gov reference data"""
    csv_files = list(Path("examples").glob(f"{nct_number}_ct_*.csv"))
    if csv_files:
        with open(csv_files[0], 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return next(reader, {})
    return {}


def create_extraction_summary(checkpoints: Dict[str, Any]) -> pd.DataFrame:
    """Create summary dataframe of all extractions"""
    summary_data = []
    
    for key, checkpoint in checkpoints.items():
        total = checkpoint['total_fields']
        completed = checkpoint['completed_fields']
        
        summary_data.append({
            'Study': key,
            'NCT Number': checkpoint['nct_number'],
            'Document Type': checkpoint['pdf_type'],
            'Fields Extracted': f"{completed}/{total}",
            'Success Rate': f"{completed/total*100:.1f}%",
            'Status': 'Complete' if checkpoint.get('is_complete', False) else 'In Progress'
        })
    
    return pd.DataFrame(summary_data)


def display_field_comparison(field_name: str, extracted: str, ctgov: str, 
                           match_status: Optional[str] = None) -> None:
    """Display side-by-side field comparison"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📄 Extracted from PDF**")
        if extracted and extracted != 'NOT_FOUND':
            st.info(extracted[:500] + "..." if len(str(extracted)) > 500 else extracted)
        else:
            st.warning("Not found in PDF")
    
    with col2:
        st.markdown("**🌐 ClinicalTrials.gov**")
        if ctgov:
            st.info(ctgov[:500] + "..." if len(str(ctgov)) > 500 else ctgov)
        else:
            st.warning("No CT.gov data")
    
    if match_status:
        if "MATCH" in match_status and "✅" in match_status:
            st.success(match_status)
        elif "MISMATCH" in match_status:
            st.error(match_status)
        else:
            st.warning(match_status)


def create_accuracy_chart(matches: int, mismatches: int, not_found: int) -> go.Figure:
    """Create accuracy visualization"""
    fig = go.Figure(data=[
        go.Bar(name='Matches', x=['Results'], y=[matches], marker_color='green'),
        go.Bar(name='Mismatches', x=['Results'], y=[mismatches], marker_color='red'),
        go.Bar(name='Not Found', x=['Results'], y=[not_found], marker_color='orange')
    ])
    
    fig.update_layout(
        title='Extraction Accuracy',
        yaxis_title='Number of Fields',
        barmode='stack',
        height=300
    )
    
    return fig


# Main UI
st.sidebar.header("📊 Navigation")

# Load data
checkpoints = load_checkpoint_data()

if not checkpoints:
    st.warning("No extraction results found. Please run the extraction pipeline first.")
    st.code("python3 run_all_trials_with_reports.py", language="bash")
else:
    # Summary view
    if st.sidebar.checkbox("Show Summary", value=True):
        st.header("📈 Extraction Summary")
        summary_df = create_extraction_summary(checkpoints)
        st.dataframe(summary_df, use_container_width=True)
    
    # Individual study selector
    st.sidebar.header("🔍 Study Details")
    selected_study = st.sidebar.selectbox(
        "Select Study",
        options=list(checkpoints.keys()),
        format_func=lambda x: f"{x} ({checkpoints[x]['completed_fields']}/{checkpoints[x]['total_fields']} fields)"
    )
    
    if selected_study:
        checkpoint = checkpoints[selected_study]
        nct_number = checkpoint['nct_number']
        pdf_type = checkpoint['pdf_type']
        
        # Load CT.gov data
        ctgov_data = load_ctgov_data(nct_number)
        
        st.header(f"📋 {nct_number} - {pdf_type}")
        
        # Progress metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Fields Extracted", checkpoint['completed_fields'])
        with col2:
            st.metric("Total Fields", checkpoint['total_fields'])
        with col3:
            st.metric("Success Rate", f"{checkpoint['completed_fields']/checkpoint['total_fields']*100:.1f}%")
        with col4:
            st.metric("PDF Pages", checkpoint.get('pdf_pages', 'N/A'))
        
        # Field mapping
        field_mapping = {
            'nct_number': 'NCT Number',
            'study_title': 'Study Title',
            'acronym': 'Acronym',
            'brief_summary': 'Brief Summary',
            'study_status': 'Study Status',
            'conditions': 'Conditions',
            'interventions': 'Interventions',
            'primary_outcome_measures': 'Primary Outcome Measures',
            'secondary_outcome_measures': 'Secondary Outcome Measures',
            'sponsor': 'Sponsor',
            'phases': 'Phases',
            'enrollment': 'Enrollment',
            'study_type': 'Study Type',
            'study_design': 'Study Design',
            'sex': 'Sex',
            'age': 'Age',
            'funder_type': 'Funder Type',
        }
        
        # Filter options
        st.subheader("🔧 Filter Options")
        col1, col2 = st.columns(2)
        with col1:
            show_matches_only = st.checkbox("Show Matches Only")
        with col2:
            show_mismatches_only = st.checkbox("Show Mismatches Only")
        
        # Field-by-field comparison
        st.subheader("📊 Field-by-Field Comparison")
        
        matches = 0
        mismatches = 0
        not_found = 0
        
        for field_name, field_data in checkpoint['fields'].items():
            if field_data['status'] == 'completed':
                extracted = field_data.get('value', '')
                ctgov_field = field_mapping.get(field_name, field_name)
                ctgov_value = ctgov_data.get(ctgov_field, '')
                
                # Determine match status
                match_status = None
                is_match = False
                is_mismatch = False
                
                if extracted and extracted != 'NOT_FOUND' and ctgov_value:
                    # Simple comparison for now
                    if extracted.lower().strip() == ctgov_value.lower().strip():
                        matches += 1
                        match_status = "✅ MATCH"
                        is_match = True
                    else:
                        mismatches += 1
                        match_status = "❌ MISMATCH"
                        is_mismatch = True
                elif not extracted or extracted == 'NOT_FOUND':
                    not_found += 1
                    match_status = "⚠️ NOT FOUND IN PDF"
                else:
                    match_status = "ℹ️ NO CT.GOV DATA"
                
                # Apply filters
                if show_matches_only and not is_match:
                    continue
                if show_mismatches_only and not is_mismatch:
                    continue
                
                with st.expander(f"**{field_name}** - {match_status}"):
                    display_field_comparison(field_name, extracted, ctgov_value, match_status)
        
        # Accuracy visualization
        st.subheader("📊 Accuracy Summary")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            total_compared = matches + mismatches + not_found
            if total_compared > 0:
                st.metric("Overall Accuracy", f"{matches/total_compared*100:.1f}%")
                st.metric("Matches", matches)
                st.metric("Mismatches", mismatches)
                st.metric("Not Found", not_found)
        
        with col2:
            if total_compared > 0:
                fig = create_accuracy_chart(matches, mismatches, not_found)
                st.plotly_chart(fig, use_container_width=True)
        
        # Unique extractions
        st.subheader("🌟 Unique Extractions (Not in CT.gov)")
        unique_count = 0
        for field_name, field_data in checkpoint['fields'].items():
            if field_data['status'] == 'completed' and field_data.get('value'):
                ctgov_field = field_mapping.get(field_name, field_name)
                if not ctgov_data.get(ctgov_field):
                    unique_count += 1
                    st.success(f"**{field_name}**: {field_data['value']}")
        
        if unique_count == 0:
            st.info("No unique fields found - all extracted fields have CT.gov equivalents")
    
    # Export options
    st.sidebar.header("💾 Export Options")
    if st.sidebar.button("Export All Results as CSV"):
        # Create export data
        export_data = []
        for key, checkpoint in checkpoints.items():
            for field_name, field_data in checkpoint['fields'].items():
                if field_data['status'] == 'completed':
                    export_data.append({
                        'NCT Number': checkpoint['nct_number'],
                        'Document Type': checkpoint['pdf_type'],
                        'Field': field_name,
                        'Extracted Value': field_data.get('value', ''),
                        'Status': field_data['status']
                    })
        
        df = pd.DataFrame(export_data)
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="extraction_results.csv",
            mime="text/csv"
        )