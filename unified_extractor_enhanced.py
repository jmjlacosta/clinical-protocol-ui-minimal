#!/usr/bin/env python3
"""
Enhanced unified extraction pipeline with intelligent CT.gov comparison
"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import subprocess
import csv
from incremental_extractor.intelligent_comparator import IntelligentComparator

class EnhancedUnifiedExtractor:
    """Extract unified data with intelligent comparison"""
    
    def __init__(self):
        self.checkpoint_dir = Path("checkpoints")
        self.results_dir = Path("results/extractions")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize intelligent comparator
        api_key = os.getenv("OPENAI_API_KEY")
        self.comparator = IntelligentComparator(api_key) if api_key else None
        
    def get_trial_documents(self, nct_number: str) -> List[Tuple[str, str]]:
        """Find all documents for a given NCT number"""
        documents = []
        examples_dir = Path("examples")
        
        # Check for different document types
        doc_types = {
            "_Prot_": "Protocol",
            "_SAP_": "SAP", 
            "_ICF_": "ICF"
        }
        
        for pattern, doc_type in doc_types.items():
            pdfs = list(examples_dir.glob(f"{nct_number}{pattern}*.pdf"))
            for pdf in pdfs:
                documents.append((str(pdf), doc_type))
                
        return documents
    
    def extract_all_documents(self, nct_number: str) -> Dict[str, Any]:
        """Extract data from all documents for a trial"""
        documents = self.get_trial_documents(nct_number)
        
        if not documents:
            print(f"No documents found for {nct_number}")
            return {}
            
        print(f"\nProcessing {nct_number} with {len(documents)} documents:")
        for doc_path, doc_type in documents:
            print(f"  - {doc_type}: {doc_path}")
        
        # Extract from each document
        all_extractions = {}
        for doc_path, doc_type in documents:
            print(f"\nExtracting from {doc_type}...")
            
            # Run extraction
            cmd = ["python3", "run_incremental_extraction.py", "--nct", nct_number, "--pdf-type", doc_type]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Load the checkpoint
                checkpoint_file = self.checkpoint_dir / f"{nct_number}_{doc_type.lower()}_checkpoint.json"
                if checkpoint_file.exists():
                    with open(checkpoint_file, 'r') as f:
                        checkpoint = json.load(f)
                        all_extractions[doc_type] = checkpoint
                        
        return all_extractions
    
    def merge_extractions(self, nct_number: str, all_extractions: Dict[str, Any]) -> Dict[str, Any]:
        """Merge extractions from multiple documents with priority rules and traceability"""
        unified = {
            "nct_number": nct_number,
            "extraction_date": datetime.now().isoformat(),
            "source_documents": list(all_extractions.keys()),
            "fields": {}
        }
        
        # Priority order for different fields
        field_priorities = {
            # Protocol is best for these
            "study_title": ["Protocol", "SAP", "ICF"],
            "conditions": ["Protocol", "SAP", "ICF"],
            "interventions": ["Protocol", "SAP", "ICF"],
            "eligibility": ["Protocol", "ICF", "SAP"],
            "sponsor": ["Protocol", "SAP", "ICF"],
            
            # SAP is best for these
            "primary_outcome_measures": ["SAP", "Protocol", "ICF"],
            "secondary_outcome_measures": ["SAP", "Protocol", "ICF"],
            "statistical_methods": ["SAP", "Protocol", "ICF"],
            "sample_size": ["SAP", "Protocol", "ICF"],
            
            # ICF might have unique patient-friendly info
            "brief_summary": ["ICF", "Protocol", "SAP"],
            
            # Default priority
            "_default": ["Protocol", "SAP", "ICF"]
        }
        
        # Collect all unique fields across documents
        all_fields = set()
        for doc_type, extraction in all_extractions.items():
            all_fields.update(extraction.get("fields", {}).keys())
        
        # For each field, find best value with traceability
        for field_name in all_fields:
            # Get priority order for this field
            priority = field_priorities.get(field_name, field_priorities["_default"])
            
            # Try each document type in priority order
            for doc_type in priority:
                if doc_type in all_extractions:
                    fields = all_extractions[doc_type].get("fields", {})
                    if field_name in fields:
                        field_data = fields[field_name]
                        if field_data.get("status") == "completed" and field_data.get("value"):
                            # Found a valid value
                            unified["fields"][field_name] = {
                                "value": field_data["value"],
                                "source_document": doc_type,
                                "source_pdf": all_extractions[doc_type]["pdf_path"],
                                "extraction_time": field_data.get("extraction_time"),
                                "confidence": field_data.get("confidence")
                            }
                            break
            
            # If no valid value found, record as not found
            if field_name not in unified["fields"]:
                unified["fields"][field_name] = {
                    "value": None,
                    "source_document": "NOT_FOUND",
                    "attempted_documents": [d for d in priority if d in all_extractions]
                }
        
        # Add extraction statistics
        total_fields = len(unified["fields"])
        extracted_fields = sum(1 for f in unified["fields"].values() if f.get("value"))
        
        unified["statistics"] = {
            "total_fields": total_fields,
            "extracted_fields": extracted_fields,
            "extraction_rate": f"{extracted_fields/total_fields*100:.1f}%" if total_fields > 0 else "0%",
            "by_document": {}
        }
        
        # Stats by document
        for doc_type in all_extractions:
            doc_fields = sum(1 for f in unified["fields"].values() 
                           if f.get("source_document") == doc_type)
            unified["statistics"]["by_document"][doc_type] = doc_fields
            
        return unified
    
    def generate_enhanced_report(self, nct_number: str, unified_data: Dict[str, Any]) -> None:
        """Generate a comprehensive report with intelligent comparison"""
        nct_dir = self.results_dir / nct_number / "unified"
        nct_dir.mkdir(parents=True, exist_ok=True)
        report_path = nct_dir / f"{nct_number}_unified_report.md"
        
        # Load CT.gov data for comparison
        ctgov_data = {}
        csv_files = list(Path("examples").glob(f"{nct_number}_ct_*.csv"))
        if csv_files:
            with open(csv_files[0], 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                ctgov_data = next(reader, {})
        
        with open(report_path, 'w') as f:
            f.write(f"# Unified Extraction Report: {nct_number}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Source Documents**: {', '.join(unified_data['source_documents'])}\n")
            f.write(f"- **Total Fields**: {unified_data['statistics']['total_fields']}\n")
            f.write(f"- **Successfully Extracted**: {unified_data['statistics']['extracted_fields']}\n")
            f.write(f"- **Extraction Rate**: {unified_data['statistics']['extraction_rate']}\n\n")
            
            # By document breakdown
            f.write("### Fields by Source Document\n\n")
            for doc_type, count in unified_data['statistics']['by_document'].items():
                f.write(f"- **{doc_type}**: {count} fields\n")
            f.write("\n")
            
            # Field results with comparison
            f.write("## Extraction Results\n\n")
            
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
                'start_date': 'Start Date',
                'sex': 'Sex',
                'age': 'Age',
                'funder_type': 'Funder Type',
                'other_ids': 'Other IDs',
                'locations': 'Locations',
            }
            
            # Count comparison results
            matches = 0
            mismatches = 0
            not_found = 0
            unique_extractions = 0
            
            for field_name, field_data in unified_data['fields'].items():
                if field_data.get('value'):
                    f.write(f"### {field_name}\n\n")
                    f.write(f"- **Extracted**: {field_data['value'][:200]}{'...' if len(str(field_data['value'])) > 200 else ''}\n")
                    f.write(f"- **Source**: {field_data['source_document']} ({Path(field_data['source_pdf']).name})\n")
                    
                    # Compare with CT.gov if available
                    ctgov_field = field_mapping.get(field_name, field_name)
                    ctgov_value = ctgov_data.get(ctgov_field, '')
                    
                    if ctgov_value:
                        f.write(f"- **CT.gov**: {ctgov_value[:200]}{'...' if len(str(ctgov_value)) > 200 else ''}\n")
                        
                        # Use intelligent comparison if available
                        if self.comparator:
                            match, confidence, explanation = self.comparator.compare_fields(
                                field_name, field_data['value'], ctgov_value
                            )
                            
                            if match:
                                matches += 1
                                f.write(f"- **Status**: MATCH ({confidence:.0%}) - {explanation}\n")
                            else:
                                mismatches += 1
                                f.write(f"- **Status**: MISMATCH ({confidence:.0%}) - {explanation}\n")
                        else:
                            # Simple comparison fallback
                            if str(field_data['value']).lower().strip() == ctgov_value.lower().strip():
                                matches += 1
                                f.write(f"- **Status**: MATCH\n")
                            else:
                                mismatches += 1
                                f.write(f"- **Status**: MISMATCH\n")
                    else:
                        unique_extractions += 1
                        f.write(f"- **Status**: UNIQUE EXTRACTION (not in CT.gov)\n")
                    
                    f.write("\n")
                else:
                    not_found += 1
            
            # Not found fields section
            f.write("## Fields Not Found\n\n")
            not_found_fields = [f for f, d in unified_data['fields'].items() if not d.get('value')]
            if not_found_fields:
                for field in not_found_fields:
                    attempted = unified_data['fields'][field].get('attempted_documents', [])
                    f.write(f"- **{field}**: Attempted in {', '.join(attempted)}\n")
            else:
                f.write("All fields were successfully extracted!\n")
            
            # Comparison statistics
            f.write("\n## Comparison Statistics\n\n")
            total_compared = matches + mismatches
            if total_compared > 0:
                f.write(f"- **Matches**: {matches} ({matches/total_compared*100:.1f}%)\n")
                f.write(f"- **Mismatches**: {mismatches} ({mismatches/total_compared*100:.1f}%)\n")
                f.write(f"- **Unique Extractions**: {unique_extractions}\n")
                f.write(f"- **Not Found**: {not_found}\n")
                f.write(f"- **Overall Accuracy**: {matches/total_compared*100:.1f}%\n")
            
            # Completeness Analysis
            f.write("\n## Completeness Analysis\n\n")
            
            # Check which fields are missing in both vs missing only in our extraction
            missing_in_both = []
            missing_only_in_extraction = []
            present_in_both = []
            extracted_but_not_in_ctgov = []
            
            for field_name in unified_data['fields'].keys():
                ctgov_field = field_mapping.get(field_name, field_name)
                ctgov_value = ctgov_data.get(ctgov_field, '')
                extracted_value = unified_data['fields'][field_name].get('value')
                
                if not extracted_value and not ctgov_value:
                    missing_in_both.append(field_name)
                elif not extracted_value and ctgov_value:
                    missing_only_in_extraction.append((field_name, ctgov_value))
                elif extracted_value and ctgov_value:
                    present_in_both.append(field_name)
                elif extracted_value and not ctgov_value:
                    extracted_but_not_in_ctgov.append((field_name, extracted_value))
            
            # Report findings
            f.write(f"### Data Coverage Summary\n\n")
            f.write(f"- **Total fields**: {len(unified_data['fields'])}\n")
            f.write(f"- **Present in both**: {len(present_in_both)} ({len(present_in_both)/len(unified_data['fields'])*100:.1f}%)\n")
            f.write(f"- **Missing in both**: {len(missing_in_both)} ({len(missing_in_both)/len(unified_data['fields'])*100:.1f}%)\n")
            f.write(f"- **Missing only in extraction**: {len(missing_only_in_extraction)} ({len(missing_only_in_extraction)/len(unified_data['fields'])*100:.1f}%)\n")
            f.write(f"- **Unique to extraction**: {len(extracted_but_not_in_ctgov)} ({len(extracted_but_not_in_ctgov)/len(unified_data['fields'])*100:.1f}%)\n")
            
            # Detail sections
            if missing_only_in_extraction:
                f.write(f"\n### Critical Misses (Present in CT.gov but not extracted)\n\n")
                f.write("These fields should have been extracted:\n\n")
                for field, ctgov_val in missing_only_in_extraction:
                    f.write(f"- **{field}**: {ctgov_val[:100]}{'...' if len(str(ctgov_val)) > 100 else ''}\n")
            
            if missing_in_both:
                f.write(f"\n### Missing in Both Sources\n\n")
                f.write("These fields are not available in either source:\n\n")
                for field in missing_in_both:
                    f.write(f"- {field}\n")
            
            if extracted_but_not_in_ctgov:
                f.write(f"\n### Unique Extractions (Added value)\n\n")
                f.write("These fields were successfully extracted but not in CT.gov:\n\n")
                for field, value in extracted_but_not_in_ctgov:
                    f.write(f"- **{field}**: {value[:100]}{'...' if len(str(value)) > 100 else ''}\n")
            
            # Document contribution summary
            f.write("\n## Document Contributions\n\n")
            for doc_type, count in unified_data['statistics']['by_document'].items():
                if count > 0:
                    f.write(f"### {doc_type}\n")
                    f.write(f"Contributed {count} fields:\n")
                    doc_fields = [f for f, d in unified_data['fields'].items() 
                                if d.get('source_document') == doc_type and d.get('value')]
                    for field in doc_fields[:10]:  # Show first 10
                        f.write(f"- {field}\n")
                    if len(doc_fields) > 10:
                        f.write(f"- ... and {len(doc_fields) - 10} more\n")
                    f.write("\n")
                
        print(f"Generated enhanced unified report: {report_path}")
        
        # Also save JSON
        json_path = nct_dir / f"{nct_number}_unified_extraction.json"
        with open(json_path, 'w') as f:
            json.dump(unified_data, f, indent=2)

def main():
    """Run enhanced unified extraction for all trials"""
    extractor = EnhancedUnifiedExtractor()
    
    # Define trials
    trials = ["NCT02454972", "NCT03927651", "NCT05826873"]
    
    print("="*80)
    print("ENHANCED UNIFIED EXTRACTION PIPELINE")
    print("="*80)
    
    for nct_number in trials:
        print(f"\n{'='*60}")
        print(f"Processing {nct_number}")
        print('='*60)
        
        # Extract from all documents
        all_extractions = extractor.extract_all_documents(nct_number)
        
        if all_extractions:
            # Merge with traceability
            unified_data = extractor.merge_extractions(nct_number, all_extractions)
            
            # Generate enhanced report
            extractor.generate_enhanced_report(nct_number, unified_data)
            
            # Print summary
            print(f"\nUnified extraction complete:")
            print(f"  - Extracted {unified_data['statistics']['extracted_fields']}/{unified_data['statistics']['total_fields']} fields")
            print(f"  - Used {len(unified_data['source_documents'])} documents")

if __name__ == "__main__":
    main()