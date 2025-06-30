"""Smart outcome extractor that mimics the original approach"""
import re
import json
import logging
from typing import List, Dict, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

class SmartOutcomeExtractor:
    """Outcome extractor that uses the same approach as the original extractor_core"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
    
    def extract_outcomes(self, text: str, outcome_type: str) -> List[str]:
        """
        Extract outcomes using the same approach as the original extractor.
        This mimics the extract_outcomes function from prompts.py
        """
        # Use the same prompt structure as the original
        outcomes_prompt = f"""
Your task is to extract all {outcome_type} outcome measures OR {outcome_type} endpoints from this clinical trial protocol.

IMPORTANT: Look for both "outcome measures" AND "endpoints" as they are often used interchangeably.

YOU MUST format your response as a valid JSON array, with each outcome as an object having these exact fields:
- outcome_measure: string (this is the name of the outcome or endpoint)
- outcome_time_frame: string (when it will be measured)
- outcome_description: string (any additional details)

Examples of what to look for:
- "Primary endpoint: Overall response rate (ORR)"
- "Secondary endpoints: Duration of response, progression-free survival"
- "Primary outcome measure: Change in tumor size at 12 weeks"

If you cannot find any {outcome_type} outcomes or endpoints, return an empty array: []

DO NOT include any explanations, apologies, or text outside the JSON array.

Here is the text:
{text[:30000]}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a protocol analyzer that helps extract structured information from clinical trial protocols. Always return valid JSON when requested, with no explanations or apologies."
                    },
                    {"role": "user", "content": outcomes_prompt}
                ],
                temperature=0.0,
                max_tokens=1000
            )
            
            outcomes_result = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                # Clean up the response to get just the JSON
                json_start = outcomes_result.find('[')
                if json_start > 0:
                    outcomes_result = outcomes_result[json_start:]
                
                json_end = outcomes_result.rfind(']')
                if json_end > 0 and len(outcomes_result) > json_end + 1:
                    outcomes_result = outcomes_result[:json_end+1]
                
                outcomes = json.loads(outcomes_result)
                
                # Format outcomes for ClinicalTrials.gov
                formatted_outcomes = []
                for outcome in outcomes:
                    if isinstance(outcome, dict):
                        measure = outcome.get('outcome_measure', '')
                        timeframe = outcome.get('outcome_time_frame', '')
                        
                        if measure:
                            if timeframe:
                                formatted_outcomes.append(f"{measure} [Time Frame: {timeframe}]")
                            else:
                                formatted_outcomes.append(measure)
                
                return formatted_outcomes
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse {outcome_type} outcomes JSON, trying alternative extraction")
                return self._fallback_extraction(text, outcome_type)
                
        except Exception as e:
            logger.error(f"Error extracting {outcome_type} outcomes: {e}")
            return []
    
    def _fallback_extraction(self, text: str, outcome_type: str) -> List[str]:
        """Fallback extraction method when JSON parsing fails"""
        try:
            # Count outcomes first
            count_prompt = f"""
How many distinct {outcome_type} outcome measures are explicitly defined in this clinical trial protocol?
Return only a single number (e.g., "3"). If none are found, return "0".
DO NOT include any other text, explanations, or apologies.

{text[:15000]}
"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": count_prompt}],
                temperature=0.0,
                max_tokens=10
            )
            
            outcome_count = response['choices'][0]['message']['content'].strip()
            outcome_count = ''.join(c for c in outcome_count if c.isdigit())
            
            if not outcome_count or outcome_count == "0":
                return []
            
            count = min(int(outcome_count), 10)  # Cap at 10 to avoid excessive API calls
            outcomes = []
            
            # Extract each outcome individually
            for i in range(1, count + 1):
                measure_prompt = f"""
What is the exact name or title of {outcome_type} outcome measure #{i} in this protocol?
Return ONLY the name/title text with no additional explanation.

{text[:15000]}
"""
                
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": measure_prompt}],
                    temperature=0.0,
                    max_tokens=100
                )
                
                measure = response['choices'][0]['message']['content'].strip()
                
                if measure and not measure.startswith("I'm sorry"):
                    # Try to get timeframe
                    timeframe_prompt = f"""
What is the time frame for the {outcome_type} outcome "{measure}" in this protocol?
Return ONLY the timeframe (e.g., "6 months", "until progression") with no explanation.
If not specified, return "Not specified".

{text[:15000]}
"""
                    
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": timeframe_prompt}],
                        temperature=0.0,
                        max_tokens=50
                    )
                    
                    timeframe = response['choices'][0]['message']['content'].strip()
                    
                    if timeframe and timeframe != "Not specified":
                        outcomes.append(f"{measure} [Time Frame: {timeframe}]")
                    else:
                        outcomes.append(measure)
            
            return outcomes
            
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            return []