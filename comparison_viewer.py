"""
Visual comparison tool for extracted vs ClinicalTrials.gov data.

Run with: streamlit run comparison_viewer.py
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any

st.set_page_config(
    page_title="Clinical Trial Extraction Comparison", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Clinical Trial Extraction Comparison Tool")
st.markdown("Compare extracted data from PDFs with ClinicalTrials.gov reference data")


def load_comparison_results(results_dir: Path) -> Dict[str, Any]:
    """Load all comparison results"""
    results = {}
    
    # Load individual comparison files
    for json_file in results_dir.glob("NCT*_comparison.json"):
        with open(json_file, 'r') as f:
            nct_id = json_file.stem.replace('_comparison', '')
            results[nct_id] = json.load(f)
    
    # Load batch summary if available
    summary_file = results_dir / 'batch_comparison_detailed.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            results['_summary'] = json.load(f)
    
    return results


def create_match_pie_chart(summary: Dict[str, Any]) -> go.Figure:
    """Create pie chart of match types"""
    labels = ['Exact Match', 'Partial Match', 'Missing', 'Different']
    values = [
        summary['exact_matches'],
        summary['partial_matches'],
        summary['missing_fields'],
        summary['different_fields']
    ]
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        marker_colors=colors
    )])
    
    fig.update_layout(
        title="Field Match Distribution",
        height=400
    )
    
    return fig


def create_confidence_histogram(comparisons: List[Dict[str, Any]]) -> go.Figure:
    """Create histogram of confidence scores"""
    confidences = [c['confidence'] for c in comparisons]
    
    fig = px.histogram(
        x=confidences,
        nbins=20,
        title="Confidence Score Distribution",
        labels={'x': 'Confidence Score', 'y': 'Number of Fields'},
        color_discrete_sequence=['#3498db']
    )
    
    fig.update_layout(height=300)
    
    return fig


def display_field_comparison(comparison: Dict[str, Any]):
    """Display a single field comparison"""
    # Color code based on match status
    status_colors = {
        'exact': '#d4edda',
        'partial': '#fff3cd',
        'missing': '#f8d7da',
        'different': '#e2e3e5'
    }
    
    status_icons = {
        'exact': '✅',
        'partial': '🔶',
        'missing': '❌',
        'different': '❓'
    }
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown(f"### {status_icons[comparison['match_status']]}")
        st.metric("Confidence", f"{comparison['confidence']:.2f}")
    
    with col2:
        with st.container():
            st.markdown(f"**Field:** {comparison['field']}")
            
            # Create expandable sections for values
            with st.expander("Show Values"):
                col_our, col_ct = st.columns(2)
                
                with col_our:
                    st.markdown("**Our Extraction:**")
                    if isinstance(comparison['our_value'], list):
                        for item in comparison['our_value']:
                            st.write(f"• {item}")
                    else:
                        st.write(comparison['our_value'] or "_Not extracted_")
                
                with col_ct:
                    st.markdown("**ClinicalTrials.gov:**")
                    if isinstance(comparison['ct_value'], list):
                        for item in comparison['ct_value']:
                            st.write(f"• {item}")
                    else:
                        st.write(comparison['ct_value'] or "_Not provided_")
            
            if comparison.get('notes'):
                st.info(f"ℹ️ {comparison['notes']}")


def main():
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    results_dir = Path(st.sidebar.text_input(
        "Results Directory",
        value="comparison_results",
        help="Directory containing comparison JSON files"
    ))
    
    if not results_dir.exists():
        st.error(f"Results directory '{results_dir}' not found!")
        st.info("Run the comparison scripts first to generate results.")
        return
    
    # Load results
    with st.spinner("Loading comparison results..."):
        results = load_comparison_results(results_dir)
    
    if not results:
        st.warning("No comparison results found in the specified directory.")
        return
    
    # Remove summary from trial list
    trial_ids = [k for k in results.keys() if k != '_summary']
    
    # Trial selector
    selected_trial = st.sidebar.selectbox(
        "Select Trial",
        options=trial_ids,
        format_func=lambda x: f"{x} ({len(results[x].get('source_pdfs', []))} PDFs)"
    )
    
    if selected_trial:
        trial_data = results[selected_trial]
        
        # Header with trial info
        st.header(f"Trial: {selected_trial}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Source PDFs", len(trial_data.get('source_pdfs', [])))
        with col2:
            st.metric("Fields Compared", trial_data['summary']['total_fields'])
        with col3:
            st.metric("Match Rate", f"{trial_data['summary']['match_rate']*100:.1f}%")
        
        # Visualization tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Field Details", "📈 Analytics", "📄 Raw Data"])
        
        with tab1:
            # Summary metrics
            st.subheader("Extraction Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart
                fig = create_match_pie_chart(trial_data['summary'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence histogram
                fig = create_confidence_histogram(trial_data['field_comparisons'])
                st.plotly_chart(fig, use_container_width=True)
            
            # Source PDFs
            st.subheader("Source Documents")
            for pdf in trial_data.get('source_pdfs', []):
                st.write(f"📄 {Path(pdf).name}")
        
        with tab2:
            # Field comparison details
            st.subheader("Field-by-Field Comparison")
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                status_filter = st.multiselect(
                    "Filter by Status",
                    options=['exact', 'partial', 'missing', 'different'],
                    default=['exact', 'partial', 'missing', 'different']
                )
            
            with col2:
                min_confidence = st.slider(
                    "Minimum Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1
                )
            
            # Sort options
            sort_by = st.selectbox(
                "Sort by",
                options=['field', 'confidence', 'match_status'],
                index=1
            )
            
            # Filter and sort comparisons
            filtered_comps = [
                c for c in trial_data['field_comparisons']
                if c['match_status'] in status_filter and c['confidence'] >= min_confidence
            ]
            
            if sort_by == 'confidence':
                filtered_comps.sort(key=lambda x: x['confidence'], reverse=True)
            elif sort_by == 'match_status':
                filtered_comps.sort(key=lambda x: x['match_status'])
            else:
                filtered_comps.sort(key=lambda x: x['field'])
            
            # Display comparisons
            for comp in filtered_comps:
                with st.container():
                    display_field_comparison(comp)
                    st.divider()
        
        with tab3:
            # Analytics
            st.subheader("Extraction Analytics")
            
            # Create DataFrame for analysis
            df = pd.DataFrame(trial_data['field_comparisons'])
            
            # Field performance chart
            field_stats = df.groupby('field').agg({
                'confidence': 'mean',
                'match_status': lambda x: (x == 'exact').sum() + (x == 'partial').sum()
            }).round(2)
            
            field_stats.columns = ['Avg Confidence', 'Successful Matches']
            field_stats = field_stats.sort_values('Avg Confidence', ascending=True)
            
            fig = px.bar(
                field_stats,
                x='Avg Confidence',
                y=field_stats.index,
                orientation='h',
                title='Field Extraction Performance',
                color='Avg Confidence',
                color_continuous_scale='RdYlGn'
            )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Unmapped fields
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Unmapped Our Fields")
                unmapped_our = trial_data.get('unmapped_our_fields', [])
                if unmapped_our:
                    for field in unmapped_our[:10]:
                        st.write(f"• {field}")
                else:
                    st.info("All fields mapped")
            
            with col2:
                st.subheader("Unmapped CT.gov Fields")
                unmapped_ct = trial_data.get('unmapped_ct_fields', [])
                if unmapped_ct:
                    for field in unmapped_ct[:10]:
                        st.write(f"• {field}")
                    if len(unmapped_ct) > 10:
                        st.write(f"... and {len(unmapped_ct) - 10} more")
                else:
                    st.info("All fields mapped")
        
        with tab4:
            # Raw data view
            st.subheader("Raw Comparison Data")
            st.json(trial_data, expanded=False)
    
    # Batch summary if available
    if '_summary' in results:
        with st.sidebar:
            st.divider()
            st.subheader("Batch Summary")
            summary = results['_summary']
            st.metric("Total Trials", summary['total_trials'])
            st.metric("Avg Match Rate", f"{summary['overall_statistics']['avg_match_rate']:.1f}%")
            
            if st.button("Show Batch Report"):
                st.session_state.show_batch = True
    
    # Show batch report in main area if requested
    if hasattr(st.session_state, 'show_batch') and st.session_state.show_batch:
        st.divider()
        st.header("Batch Comparison Report")
        
        summary = results['_summary']
        
        # Overall stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trials", summary['total_trials'])
        with col2:
            st.metric("Successful", summary['successful'])
        with col3:
            st.metric("Failed", summary['failed'])
        with col4:
            st.metric("Avg Match Rate", f"{summary['overall_statistics']['avg_match_rate']:.1f}%")
        
        # Best/worst performing fields
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Best Performing Fields")
            for field, stats in summary['field_statistics']['best_fields'].items():
                st.write(f"**{field}**: {stats['success_rate']*100:.1f}% success")
        
        with col2:
            st.subheader("Worst Performing Fields")
            for field, stats in summary['field_statistics']['worst_fields'].items():
                st.write(f"**{field}**: {stats['success_rate']*100:.1f}% success")


if __name__ == "__main__":
    main()