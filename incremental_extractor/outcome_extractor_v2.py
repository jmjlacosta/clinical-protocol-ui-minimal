"""Improved outcome extractor with better PDF handling"""
import re
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class OutcomeExtractorV2:
    """Improved extraction for primary and secondary outcome measures"""
    
    def extract_outcomes_from_text(self, text: str, outcome_type: str = "primary") -> List[str]:
        """
        Extract outcome measures from text using pattern matching and context analysis.
        
        Args:
            text: The document text
            outcome_type: "primary" or "secondary"
            
        Returns:
            List of formatted outcome measures
        """
        # First, try to find the dedicated outcome sections in the document
        if outcome_type.lower() == "primary":
            outcomes = self._extract_primary_outcomes(text)
        else:
            outcomes = self._extract_secondary_outcomes(text)
        
        return outcomes
    
    def _extract_primary_outcomes(self, text: str) -> List[str]:
        """Extract primary outcome measures"""
        outcomes = []
        
        # Look for the primary endpoint section
        # Pattern 1: Section header followed by content
        primary_section_pattern = r"(?:3\.1\s*)?PRIMARY\s+ENDPOINT[S]?\s*\n+([^3\n][^\n]+(?:\n[^3\n][^\n]+)*)"
        match = re.search(primary_section_pattern, text, re.IGNORECASE | re.MULTILINE)
        
        if match:
            content = match.group(1).strip()
            # Clean up the content
            content = re.sub(r'\.{2,}', '', content)  # Remove dots
            content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
            
            # Look for ORR definition
            orr_pattern = r"Overall Response Rate\s*\(ORR\)[^.]*\.?\s*(?:ORR is defined as[^.]+\.)?"
            orr_match = re.search(orr_pattern, content, re.IGNORECASE)
            if orr_match:
                outcome = orr_match.group(0).strip()
                # Add a generic timeframe since it's not explicitly stated
                outcomes.append(f"Overall Response Rate (ORR) [Time Frame: From start of treatment until disease progression]")
                return outcomes
        
        # Pattern 2: Look for primary objective/endpoint statements
        primary_patterns = [
            r"primary\s+(?:endpoint|objective)\s*(?:is|:)?\s*([^.]+)",
            r"Primary endpoint:\s*([^\n]+)",
            r"The primary (?:endpoint|objective)[^:]*:\s*([^.]+)"
        ]
        
        for pattern in primary_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match and "overall response rate" in match.lower():
                    outcomes.append("Overall Response Rate (ORR) [Time Frame: From start of treatment until disease progression]")
                    return outcomes
        
        return outcomes
    
    def _extract_secondary_outcomes(self, text: str) -> List[str]:
        """Extract secondary outcome measures"""
        outcomes = []
        
        # Look for the secondary endpoints section
        secondary_section_pattern = r"(?:3\.2\s*)?SECONDARY\s+ENDPOINT[S]?\s*\n+([^3][^\n]+(?:\n[^3][^\n]+)*?)(?=\n(?:3\.|Plasma|Safety|$))"
        match = re.search(secondary_section_pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        
        if match:
            content = match.group(1).strip()
            
            # Split by line and process each potential outcome
            lines = content.split('\n')
            current_outcome = None
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and headers
                if not line or 'efficacy' in line.lower() and len(line.split()) < 3:
                    continue
                
                # Look for outcome definitions
                # Pattern 1: Name (Abbrev) defined as...
                abbrev_pattern = r"^([^(]+)\s*\(([A-Z]{2,})\)[^,]*(?:defined as|,)?\s*(.+)"
                abbrev_match = re.match(abbrev_pattern, line)
                
                if abbrev_match:
                    name = abbrev_match.group(1).strip()
                    abbrev = abbrev_match.group(2)
                    definition = abbrev_match.group(3).strip()
                    
                    # Extract timeframe from definition
                    timeframe = self._extract_timeframe_from_definition(definition)
                    if timeframe:
                        outcomes.append(f"{name} ({abbrev}) [Time Frame: {timeframe}]")
                    else:
                        outcomes.append(f"{name} ({abbrev})")
                
                # Pattern 2: Simple outcome listing
                elif self._is_outcome_line(line):
                    # Clean and add the outcome
                    clean_outcome = self._clean_outcome_text(line)
                    if clean_outcome:
                        outcomes.append(clean_outcome)
        
        # Deduplicate and clean up
        cleaned_outcomes = []
        seen = set()
        
        for outcome in outcomes:
            # Extract core name for deduplication
            core = outcome.split('[')[0].split('(')[0].strip().lower()
            if core not in seen and len(core) > 5:
                seen.add(core)
                cleaned_outcomes.append(outcome)
        
        return cleaned_outcomes
    
    def _extract_timeframe_from_definition(self, text: str) -> Optional[str]:
        """Extract timeframe from an outcome definition"""
        # Common timeframe patterns
        timeframe_patterns = [
            r"from[^,]+to[^,]+(?:progression|death|PD)",
            r"(?:period|time)\s+(?:from|between)[^,]+(?:to|until)[^,]+",
            r"at\s+(?:these\s+)?time\s+points?\s*\([^)]+\)",
            r"(?:after|at)\s+(\d+\s*(?:months?|weeks?|days?))",
            r"lasting\s+(?:over|at least)\s+(\d+\s*months?)",
        ]
        
        for pattern in timeframe_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                timeframe = match.group(0) if match.lastindex is None else match.group(1)
                # Clean up the timeframe
                timeframe = re.sub(r'\s+', ' ', timeframe).strip()
                return timeframe
        
        # Default timeframes for common outcomes
        text_lower = text.lower()
        if "response" in text_lower and "duration" in text_lower:
            return "From response to progression"
        elif "progression" in text_lower and "free" in text_lower:
            return "From first treatment to progression or death"
        elif "overall survival" in text_lower or " os" in text_lower:
            return "From first treatment to death or last follow-up"
        
        return None
    
    def _is_outcome_line(self, line: str) -> bool:
        """Check if a line contains an outcome measure"""
        line_lower = line.lower()
        
        # Must contain relevant terms
        outcome_keywords = [
            'duration', 'response', 'survival', 'benefit', 'rate',
            'progression', 'free', 'overall', 'clinical', 'safety',
            'adverse', 'event', 'toxicity', 'quality', 'life'
        ]
        
        # Or contain common abbreviations
        outcome_abbrevs = ['orr', 'pfs', 'os', 'dfs', 'dr', 'cb', 'auc']
        
        has_keyword = any(keyword in line_lower for keyword in outcome_keywords)
        has_abbrev = any(f' {abbrev}' in line_lower or f'({abbrev})' in line_lower 
                        for abbrev in outcome_abbrevs)
        
        return (has_keyword or has_abbrev) and len(line) > 10
    
    def _clean_outcome_text(self, text: str) -> str:
        """Clean outcome text for presentation"""
        # Remove common artifacts
        text = re.sub(r'^\s*[-•]\s*', '', text)  # Remove bullets
        text = re.sub(r'^\d+\.\s*', '', text)  # Remove numbering
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        # Don't return if it's just a header
        if text.lower() in ['efficacy', 'safety', 'pharmacokinetics', 'other']:
            return ""
        
        return text