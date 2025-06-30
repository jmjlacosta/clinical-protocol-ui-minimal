"""Generate comparison reports in various formats"""
import os
import csv
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from .schema import StudyComparison, ExtractionCheckpoint

class ReportGenerator:
    """Generate reports from extraction and comparison results"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_csv_report(self, comparisons: List[StudyComparison], filename: str = "comparison_report.csv") -> str:
        """Generate CSV report of comparisons"""
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            # Define columns
            fieldnames = [
                'nct_number', 'pdf_type', 'field_name', 
                'extracted_value', 'ctgov_value', 
                'match', 'similarity_score', 'notes'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write data for each comparison
            for comparison in comparisons:
                for field_comp in comparison.field_comparisons:
                    writer.writerow({
                        'nct_number': comparison.nct_number,
                        'pdf_type': comparison.pdf_type,
                        'field_name': field_comp.field_name,
                        'extracted_value': field_comp.extracted_value or '',
                        'ctgov_value': field_comp.ctgov_value or '',
                        'match': 'Yes' if field_comp.match else 'No',
                        'similarity_score': field_comp.similarity_score or 0,
                        'notes': field_comp.notes or ''
                    })
        
        return output_path
    
    def generate_summary_report(self, comparisons: List[StudyComparison], 
                               filename: str = "summary_report.txt") -> str:
        """Generate human-readable summary report"""
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Clinical Trial Data Extraction Comparison Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Studies: {len(comparisons)}\n\n")
            
            # Overall statistics
            total_fields = sum(c.total_fields for c in comparisons)
            total_matches = sum(c.matching_fields for c in comparisons)
            overall_match_rate = (total_matches / total_fields * 100) if total_fields > 0 else 0
            
            f.write("Overall Statistics:\n")
            f.write(f"  Total Fields Compared: {total_fields}\n")
            f.write(f"  Total Matches: {total_matches}\n")
            f.write(f"  Overall Match Rate: {overall_match_rate:.1f}%\n\n")
            
            # Per-study summaries
            for comparison in comparisons:
                f.write("-" * 60 + "\n")
                f.write(f"Study: {comparison.nct_number} ({comparison.pdf_type})\n")
                f.write(f"PDF: {comparison.pdf_path}\n")
                f.write(f"Match Rate: {comparison.match_percentage:.1f}%\n")
                f.write(f"  Matching Fields: {comparison.matching_fields}/{comparison.total_fields}\n")
                f.write(f"  Mismatched Fields: {comparison.mismatched_fields}\n")
                f.write(f"  Missing in Extraction: {comparison.missing_in_extraction}\n")
                f.write(f"  Missing in CT.gov: {comparison.missing_in_ctgov}\n")
                
                # Show mismatched fields
                if comparison.mismatched_fields > 0:
                    f.write("\n  Mismatched Fields:\n")
                    for field_comp in comparison.field_comparisons:
                        if not field_comp.match and field_comp.extracted_value and field_comp.ctgov_value:
                            f.write(f"    - {field_comp.field_name}:\n")
                            f.write(f"      Extracted: {field_comp.extracted_value[:100]}{'...' if len(field_comp.extracted_value or '') > 100 else ''}\n")
                            f.write(f"      CT.gov:    {field_comp.ctgov_value[:100]}{'...' if len(field_comp.ctgov_value or '') > 100 else ''}\n")
                
                f.write("\n")
        
        return output_path
    
    def generate_json_report(self, comparisons: List[StudyComparison], 
                            filename: str = "comparison_report.json") -> str:
        """Generate JSON report for programmatic access"""
        output_path = os.path.join(self.output_dir, filename)
        
        # Convert to dict for JSON serialization
        report_data = {
            "generated": datetime.now().isoformat(),
            "total_studies": len(comparisons),
            "comparisons": []
        }
        
        for comparison in comparisons:
            comp_dict = comparison.model_dump()
            # Convert datetime to string
            comp_dict['comparison_time'] = comp_dict['comparison_time'].isoformat()
            report_data['comparisons'].append(comp_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        
        return output_path
    
    def generate_field_accuracy_report(self, comparisons: List[StudyComparison],
                                     filename: str = "field_accuracy_report.csv") -> str:
        """Generate report showing accuracy by field across all studies"""
        output_path = os.path.join(self.output_dir, filename)
        
        # Aggregate statistics by field
        field_stats = {}
        
        for comparison in comparisons:
            for field_comp in comparison.field_comparisons:
                field_name = field_comp.field_name
                
                if field_name not in field_stats:
                    field_stats[field_name] = {
                        'total': 0,
                        'matches': 0,
                        'mismatches': 0,
                        'missing_extraction': 0,
                        'missing_ctgov': 0
                    }
                
                field_stats[field_name]['total'] += 1
                
                if not field_comp.extracted_value:
                    field_stats[field_name]['missing_extraction'] += 1
                elif not field_comp.ctgov_value:
                    field_stats[field_name]['missing_ctgov'] += 1
                elif field_comp.match:
                    field_stats[field_name]['matches'] += 1
                else:
                    field_stats[field_name]['mismatches'] += 1
        
        # Write to CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['field_name', 'total_comparisons', 'matches', 'accuracy_pct',
                         'mismatches', 'missing_extraction', 'missing_ctgov']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for field_name, stats in sorted(field_stats.items()):
                accuracy = (stats['matches'] / stats['total'] * 100) if stats['total'] > 0 else 0
                
                writer.writerow({
                    'field_name': field_name,
                    'total_comparisons': stats['total'],
                    'matches': stats['matches'],
                    'accuracy_pct': f"{accuracy:.1f}",
                    'mismatches': stats['mismatches'],
                    'missing_extraction': stats['missing_extraction'],
                    'missing_ctgov': stats['missing_ctgov']
                })
        
        return output_path
    
    def generate_checkpoint_status_report(self, checkpoint_manager: CheckpointManager,
                                        filename: str = "checkpoint_status_report.txt") -> str:
        """Generate report of all checkpoint statuses"""
        output_path = os.path.join(self.output_dir, filename)
        
        checkpoints = checkpoint_manager.list_checkpoints()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Checkpoint Status Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Checkpoints: {len(checkpoints)}\n\n")
            
            # Group by status
            complete = []
            incomplete = []
            
            for key, info in checkpoints.items():
                if info['progress'] >= 100:
                    complete.append((key, info))
                else:
                    incomplete.append((key, info))
            
            # Write complete extractions
            f.write(f"Complete Extractions ({len(complete)}):\n")
            for key, info in sorted(complete):
                f.write(f"  ✓ {key}: {info['completed_fields']}/{info['total_fields']} fields\n")
                f.write(f"    Last updated: {info['last_update'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            f.write(f"\nIncomplete Extractions ({len(incomplete)}):\n")
            for key, info in sorted(incomplete, key=lambda x: x[1]['progress'], reverse=True):
                f.write(f"  ⋯ {key}: {info['progress']:.1f}% ({info['completed_fields']}/{info['total_fields']} fields)\n")
                f.write(f"    Last updated: {info['last_update'].strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return output_path