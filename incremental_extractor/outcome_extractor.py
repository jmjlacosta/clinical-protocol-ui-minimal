"""Specialized extractor for outcome measures"""
import re
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class OutcomeExtractor:
    """Specialized extraction for primary and secondary outcome measures"""
    
    def extract_outcomes_from_text(self, text: str, outcome_type: str = "primary") -> List[str]:
        """
        Extract outcome measures from text using pattern matching and context analysis.
        
        Args:
            text: The document text
            outcome_type: "primary" or "secondary"
            
        Returns:
            List of formatted outcome measures
        """
        outcomes = []
        
        # Step 1: Find relevant sections
        sections = self._find_outcome_sections(text, outcome_type)
        
        # Step 2: Extract outcomes from each section
        for section in sections:
            extracted = self._extract_from_section(section, outcome_type)
            outcomes.extend(extracted)
        
        # Step 3: Deduplicate and format
        unique_outcomes = self._deduplicate_outcomes(outcomes)
        
        return unique_outcomes
    
    def _find_outcome_sections(self, text: str, outcome_type: str) -> List[str]:
        """Find text sections that contain outcome definitions"""
        sections = []
        lines = text.split('\n')
        
        # Define search patterns
        if outcome_type.lower() == "primary":
            patterns = [
                r"primary\s+endpoint",
                r"primary\s+outcome",
                r"primary\s+objective",
                r"3\.1\s+primary",
                r"primary\s+efficacy"
            ]
        else:
            patterns = [
                r"secondary\s+endpoint",
                r"secondary\s+outcome",
                r"secondary\s+objective",
                r"3\.2\s+secondary",
                r"secondary\s+efficacy"
            ]
        
        # Search for patterns
        for i, line in enumerate(lines):
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Extract context (more for secondary as there are usually more items)
                    context_size = 30 if outcome_type == "primary" else 50
                    start = max(0, i - 2)
                    end = min(len(lines), i + context_size)
                    section = '\n'.join(lines[start:end])
                    sections.append(section)
                    break
        
        return sections
    
    def _extract_from_section(self, section: str, outcome_type: str) -> List[str]:
        """Extract outcome measures from a text section"""
        outcomes = []
        
        # Common outcome patterns
        outcome_patterns = [
            # Pattern 1: Outcome name followed by definition
            r"(?:^|\n)\s*[-•]\s*([^:\n]+?)(?:\s*[:]\s*([^\n]+))?",
            # Pattern 2: Outcome abbreviation pattern
            r"(\w+(?:\s+\w+)*)\s*\(([A-Z]{2,})\)[^:]*?(?:defined as|is|:|measured)",
            # Pattern 3: Simple listing
            r"(?:^|\n)\s*(?:\d+\.|[-•])\s*([^\n]+)",
        ]
        
        # Extract primary endpoint if it's clearly stated
        if outcome_type == "primary":
            # Look for direct statements
            primary_match = re.search(
                r"(?:primary\s+endpoint|primary\s+outcome)[:\s]+([^\n.]+?)(?:\.|$)",
                section, re.IGNORECASE
            )
            if primary_match:
                outcome = primary_match.group(1).strip()
                # Look for timeframe
                timeframe = self._extract_timeframe(section, outcome)
                if timeframe:
                    outcomes.append(f"{outcome} [Time Frame: {timeframe}]")
                else:
                    outcomes.append(outcome)
                return outcomes
        
        # Extract outcomes using patterns
        extracted_items = []
        
        # Special handling for sections that list outcomes
        if outcome_type == "secondary":
            # Look for structured lists of secondary outcomes
            lines = section.split('\n')
            current_outcome = None
            
            for line in lines:
                line = line.strip()
                
                # Skip headers and empty lines
                if not line or 'secondary' in line.lower() and 'endpoint' in line.lower():
                    continue
                
                # Check if this is an outcome definition
                outcome_indicators = ['defined as', 'measured', 'calculated', 'rate of', 'time to', 'proportion']
                abbreviation_match = re.match(r"^(.+?)\s*\(([A-Z]{2,})\)", line)
                
                if abbreviation_match or any(indicator in line.lower() for indicator in outcome_indicators):
                    if abbreviation_match:
                        full_name = abbreviation_match.group(1).strip()
                        abbrev = abbreviation_match.group(2)
                        
                        # Get the rest of the line for timeframe info
                        rest_of_line = line[abbreviation_match.end():].strip()
                        timeframe = self._extract_timeframe_from_definition(rest_of_line)
                        
                        if timeframe:
                            extracted_items.append(f"{full_name} ({abbrev}) [Time Frame: {timeframe}]")
                        else:
                            extracted_items.append(f"{full_name} ({abbrev})")
                    else:
                        # Extract outcome name and look for timeframe
                        outcome_name = self._clean_outcome_name(line)
                        if outcome_name:
                            extracted_items.append(outcome_name)
        
        # If we have extracted items, use them
        if extracted_items:
            outcomes = extracted_items
        else:
            # Fallback to general extraction
            for pattern in outcome_patterns:
                matches = re.findall(pattern, section, re.MULTILINE)
                for match in matches:
                    if isinstance(match, tuple):
                        outcome = ' '.join(m.strip() for m in match if m.strip())
                    else:
                        outcome = match.strip()
                    
                    if self._is_valid_outcome(outcome):
                        outcomes.append(outcome)
        
        return outcomes
    
    def _extract_timeframe(self, text: str, outcome_name: str) -> Optional[str]:
        """Extract timeframe for a specific outcome"""
        # Common timeframe patterns
        timeframe_patterns = [
            r"assessed?\s+(?:every|at|after)\s+([^,.]+)",
            r"measured?\s+(?:every|at|after)\s+([^,.]+)",
            r"evaluated?\s+(?:every|at|after)\s+([^,.]+)",
            r"from\s+[\w\s]+to\s+([^,.]+)",
            r"until\s+([^,.]+)",
            r"(?:at|after)\s+(\d+\s*(?:weeks?|months?|years?|days?))",
            r"(?:every)\s+(\d+\s*(?:weeks?|months?|years?|days?|cycles?))",
        ]
        
        # Search near the outcome name
        for pattern in timeframe_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_timeframe_from_definition(self, text: str) -> Optional[str]:
        """Extract timeframe from an outcome definition line"""
        # Look for common timeframe indicators
        timeframe_match = re.search(
            r"(?:from|between|at|after|every|until|through)\s+([^,.]+?)(?:\.|,|$)",
            text, re.IGNORECASE
        )
        if timeframe_match:
            return timeframe_match.group(1).strip()
        
        # Look for specific time periods
        period_match = re.search(
            r"(\d+\s*(?:weeks?|months?|years?|days?|cycles?))",
            text, re.IGNORECASE
        )
        if period_match:
            return period_match.group(1).strip()
        
        return None
    
    def _clean_outcome_name(self, text: str) -> Optional[str]:
        """Clean and validate an outcome name"""
        # Remove common prefixes
        text = re.sub(r"^[-•·]\s*", "", text).strip()
        text = re.sub(r"^\d+\.\s*", "", text).strip()
        
        # Skip if too short or contains only stop words
        if len(text) < 5 or text.lower() in ['the', 'and', 'or', 'of', 'in', 'at']:
            return None
        
        # Skip headers
        if any(word in text.lower() for word in ['endpoint', 'outcome', 'measure', 'objective', 'efficacy']):
            if len(text.split()) <= 3:  # Just a header
                return None
        
        return text
    
    def _is_valid_outcome(self, outcome: str) -> bool:
        """Check if the extracted text is a valid outcome measure"""
        # Must have minimum length
        if len(outcome) < 5:
            return False
        
        # Should not be just a header
        header_words = ['endpoint', 'outcome', 'measure', 'objective', 'section']
        if any(outcome.lower().strip() == word for word in header_words):
            return False
        
        # Should contain relevant terms
        relevant_terms = [
            'rate', 'survival', 'response', 'progression', 'free', 'overall',
            'duration', 'time', 'benefit', 'quality', 'safety', 'toxicity',
            'adverse', 'event', 'orr', 'pfs', 'os', 'dfs', 'dr', 'cb'
        ]
        
        outcome_lower = outcome.lower()
        return any(term in outcome_lower for term in relevant_terms)
    
    def _deduplicate_outcomes(self, outcomes: List[str]) -> List[str]:
        """Remove duplicate outcomes while preserving the most complete version"""
        if not outcomes:
            return []
        
        # Sort by length (longer is usually more complete)
        outcomes.sort(key=len, reverse=True)
        
        unique = []
        seen_cores = set()
        
        for outcome in outcomes:
            # Extract core outcome name (before brackets)
            core = outcome.split('[')[0].strip()
            core_lower = core.lower()
            
            # Check if we've seen a similar outcome
            is_duplicate = False
            for seen in seen_cores:
                if seen in core_lower or core_lower in seen:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(outcome)
                seen_cores.add(core_lower)
        
        return unique