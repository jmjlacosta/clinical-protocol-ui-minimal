"""
Enhanced Streamlit UI for Clinical Trial Extraction with Comparison
"""
import streamlit as st
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import os
from typing import List, Dict, Any
import uuid
from pipeline_adapter import PipelineAdapter, ExtractionJob

# Page config
st.set_page_config(
    page_title="Clinical Protocol Extractor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'jobs' not in st.session_state:
    st.session_state.jobs = {}
if 'current_job_id' not in st.session_state:
    st.session_state.current_job_id = None

# Initialize pipeline adapter
@st.cache_resource
def get_pipeline_adapter():
    return PipelineAdapter()

def create_progress_placeholder():
    """Create a placeholder for progress updates"""
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    return progress_bar, status_text

def update_progress(progress_bar, status_text, progress: float, message: str):
    """Update progress indicators"""
    progress_bar.progress(int(progress))
    status_text.text(message)

def render_comparison_metrics(stats: Dict[str, Any]):
    """Render comparison metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Extraction Rate",
            f"{stats.get('extraction_rate', '0%')}",
            help="Percentage of fields successfully extracted"
        )
    
    with col2:
        st.metric(
            "Match Rate",
            f"{stats.get('match_rate', 0):.1f}%",
            help="Percentage of extracted fields matching CT.gov"
        )
    
    with col3:
        st.metric(
            "Total Fields",
            stats.get('total_fields', 0),
            delta=f"{stats.get('extracted_fields', 0)} extracted"
        )
    
    with col4:
        st.metric(
            "Unique Findings",
            stats.get('unique_extractions', 0),
            help="Fields found in PDFs but not in CT.gov"
        )

def render_field_comparison(unified_data: Dict[str, Any], ctgov_data: Dict[str, Any]):
    """Render detailed field comparison"""
    if 'fields' not in unified_data:
        st.warning("No field data available for comparison")
        return
    
    # Prepare comparison data
    comparison_data = []
    
    field_mapping = {
        'nct_number': 'NCT Number',
        'study_title': 'Study Title',
        'acronym': 'Acronym',
        'brief_summary': 'Brief Summary',
        'conditions': 'Conditions',
        'interventions': 'Interventions',
        'sponsor': 'Sponsor',
        'primary_outcome_measures': 'Primary Outcome Measures',
        'secondary_outcome_measures': 'Secondary Outcome Measures',
        # Add more mappings as needed
    }
    
    for field_name, field_data in unified_data['fields'].items():
        extracted_value = field_data.get('value', '')
        source_doc = field_data.get('source_document', 'N/A')
        
        # Get CT.gov value
        ctgov_field = field_mapping.get(field_name, field_name)
        ctgov_value = ctgov_data.get(ctgov_field, '') if ctgov_data else ''
        
        # Determine match status
        if not extracted_value:
            status = 'Not Found'
            status_color = '🔴'
        elif not ctgov_value:
            status = 'Unique'
            status_color = '🟡'
        elif str(extracted_value).lower().strip() == str(ctgov_value).lower().strip():
            status = 'Match'
            status_color = '🟢'
        else:
            status = 'Mismatch'
            status_color = '🟠'
        
        comparison_data.append({
            'Field': field_name,
            'Status': f"{status_color} {status}",
            'Extracted': str(extracted_value)[:100] + '...' if len(str(extracted_value)) > 100 else str(extracted_value),
            'CT.gov': str(ctgov_value)[:100] + '...' if len(str(ctgov_value)) > 100 else str(ctgov_value),
            'Source': source_doc
        })
    
    # Display as dataframe
    df = pd.DataFrame(comparison_data)
    
    # Add filters
    col1, col2 = st.columns([1, 3])
    with col1:
        status_filter = st.multiselect(
            "Filter by status",
            options=['🟢 Match', '🟠 Mismatch', '🟡 Unique', '🔴 Not Found'],
            default=['🟢 Match', '🟠 Mismatch', '🟡 Unique', '🔴 Not Found']
        )
    
    # Apply filters
    if status_filter:
        df = df[df['Status'].isin(status_filter)]
    
    # Display table
    st.dataframe(
        df,
        use_container_width=True,
        height=400,
        hide_index=True
    )
    
    # Download comparison
    csv = df.to_csv(index=False)
    st.download_button(
        "Download Comparison CSV",
        data=csv,
        file_name=f"comparison_{unified_data.get('nct_number', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def render_document_contributions(stats: Dict[str, Any]):
    """Render document contribution summary"""
    by_doc = stats.get('by_document', {})
    
    if by_doc:
        st.subheader("Field Contributions by Document Type")
        for doc_type, count in by_doc.items():
            percentage = (count / sum(by_doc.values())) * 100 if sum(by_doc.values()) > 0 else 0
            st.metric(doc_type, f"{count} fields", f"{percentage:.1f}%")

def main():
    st.title("🔬 Clinical Protocol Extractor")
    st.markdown("Extract and compare clinical trial data from PDFs with CT.gov")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # API Key is provided by HealthUniverse system
        api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key:
            st.success("✅ API Key detected")
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            st.warning("⚠️ No API key detected")
        
        # Pipeline selection
        pipeline_type = st.radio(
            "Extraction Pipeline",
            ["enhanced", "legacy"],
            format_func=lambda x: "Enhanced (with CT.gov comparison)" if x == "enhanced" else "Legacy (basic extraction)",
            help="Enhanced pipeline requires OpenAI API key"
        )
        
        # NCT Number input
        nct_number = st.text_input(
            "NCT Number",
            placeholder="NCT12345678",
            help="Will be auto-detected from filename if not provided"
        )
        
        st.divider()
        
        # Job history
        st.header("Extraction History")
        if st.session_state.jobs:
            for job_id, job in st.session_state.jobs.items():
                status_icon = {
                    'pending': '⏳',
                    'running': '🔄',
                    'completed': '✅',
                    'failed': '❌'
                }.get(job.status, '❓')
                
                if st.button(f"{status_icon} {job.nct_number} ({len(job.pdf_files)} files)", key=job_id):
                    st.session_state.current_job_id = job_id
    
    # Main content
    uploaded_files = st.file_uploader(
        "Upload Clinical Trial PDFs",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDFs (Protocol, SAP, ICF)"
    )
    
    if uploaded_files:
        st.info(f"📄 {len(uploaded_files)} file(s) uploaded")
        
        # Display uploaded files
        file_info = []
        adapter = get_pipeline_adapter()
        
        for file in uploaded_files:
            detected_nct = adapter.extract_nct_number(file.name)
            doc_type = adapter.determine_doc_type(file.name)
            
            file_info.append({
                'Filename': file.name,
                'Size': f"{file.size / 1024:.1f} KB",
                'NCT Number': detected_nct or nct_number or 'Unknown',
                'Document Type': doc_type
            })
        
        df_files = pd.DataFrame(file_info)
        st.dataframe(df_files, use_container_width=True, hide_index=True)
        
        # Extract button
        if st.button("🚀 Start Extraction", type="primary"):
            # Validate inputs
            if pipeline_type == "enhanced" and not api_key:
                st.error("OpenAI API key is required for enhanced extraction")
                return
            
            # Determine NCT number
            final_nct = nct_number
            if not final_nct:
                # Try to get from first file
                for info in file_info:
                    if info['NCT Number'] != 'Unknown':
                        final_nct = info['NCT Number']
                        break
            
            if not final_nct:
                st.error("Could not determine NCT number. Please provide it manually.")
                return
            
            # Create job
            job_id = str(uuid.uuid4())
            pdf_files = []
            
            for file, info in zip(uploaded_files, file_info):
                pdf_bytes = file.read()
                doc_type = info['Document Type']
                pdf_files.append((file.name, pdf_bytes, doc_type))
            
            job = ExtractionJob(
                job_id=job_id,
                nct_number=final_nct,
                pdf_files=pdf_files,
                pipeline_type=pipeline_type
            )
            
            st.session_state.jobs[job_id] = job
            st.session_state.current_job_id = job_id
            
            # Run extraction with progress
            progress_bar, status_text = create_progress_placeholder()
            
            def progress_callback(progress: float, message: str):
                update_progress(progress_bar, status_text, progress, message)
            
            # Process job
            adapter = get_pipeline_adapter()
            job = adapter.process_job(job, progress_callback)
            
            # Update in session state
            st.session_state.jobs[job_id] = job
            
            # Show results
            if job.status == 'completed':
                st.success("✅ Extraction completed successfully!")
                st.rerun()
            else:
                st.error(f"❌ Extraction failed: {job.error}")
    
    # Display results for current job
    if st.session_state.current_job_id:
        job = st.session_state.jobs.get(st.session_state.current_job_id)
        
        if job and job.status == 'completed':
            st.divider()
            st.header(f"Results: {job.nct_number}")
            
            if job.pipeline_type == 'enhanced' and 'unified_data' in job.results:
                # Enhanced pipeline results
                unified_data = job.results['unified_data']
                ctgov_data = job.results.get('ctgov_data', {})
                
                # Tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Field Comparison", "📈 Statistics", "📄 Reports"])
                
                with tab1:
                    # Show metrics
                    adapter = get_pipeline_adapter()
                    stats = adapter.get_comparison_stats(unified_data, ctgov_data)
                    render_comparison_metrics(stats)
                    
                    # Document contributions
                    render_document_contributions(stats)
                
                with tab2:
                    # Detailed field comparison
                    render_field_comparison(unified_data, ctgov_data)
                
                with tab3:
                    # Statistics and insights
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Extraction Statistics")
                        st.json(unified_data.get('statistics', {}))
                    
                    with col2:
                        st.subheader("Source Documents")
                        for doc in unified_data.get('source_documents', []):
                            st.write(f"• {doc}")
                
                with tab4:
                    # Reports and downloads
                    st.subheader("Generated Reports")
                    
                    # Report path
                    report_path = job.results.get('report_path')
                    if report_path and Path(report_path).exists():
                        with open(report_path, 'r') as f:
                            report_content = f.read()
                        
                        st.download_button(
                            "📄 Download Full Report (Markdown)",
                            data=report_content,
                            file_name=f"{job.nct_number}_report.md",
                            mime="text/markdown"
                        )
                        
                        # Show preview
                        with st.expander("Report Preview"):
                            st.markdown(report_content[:2000] + "..." if len(report_content) > 2000 else report_content)
                    
                    # JSON download
                    st.download_button(
                        "📋 Download Extraction Data (JSON)",
                        data=json.dumps(unified_data, indent=2),
                        file_name=f"{job.nct_number}_extraction.json",
                        mime="application/json"
                    )
            
            else:
                # Legacy pipeline results
                st.subheader("Extraction Results")
                
                for filename, result in job.results.items():
                    with st.expander(f"📄 {filename}"):
                        if result['status'] == 'success':
                            extraction = result['extraction']
                            
                            # Display key fields
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Brief Title:**", extraction.get('brief_title', 'N/A'))
                                st.write("**Conditions:**", extraction.get('conditions', 'N/A'))
                                st.write("**Sponsor:**", extraction.get('sponsor', 'N/A'))
                            
                            with col2:
                                st.write("**Study Type:**", extraction.get('study_type', 'N/A'))
                                st.write("**Phase:**", extraction.get('phase', 'N/A'))
                                st.write("**Enrollment:**", extraction.get('enrollment', 'N/A'))
                            
                            # Full JSON
                            st.json(extraction, expanded=False)
                            
                            # Download
                            st.download_button(
                                f"Download {filename} JSON",
                                data=json.dumps(extraction, indent=2),
                                file_name=f"{Path(filename).stem}_extraction.json",
                                mime="application/json",
                                key=f"download_{filename}"
                            )
                        else:
                            st.error(f"Extraction failed: {result['error']}")

if __name__ == "__main__":
    main()