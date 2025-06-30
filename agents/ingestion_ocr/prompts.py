import os
import json
import logging
from typing import Dict, Any, List
import openai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
if not api_key:
    raise ValueError("Neither OPENAI_API_KEY nor API_KEY environment variable is set")
openai.api_key = api_key


def query_gpt(prompt: str, model: str = "gpt-4o", temperature: float = 0.0) -> str:
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": """You are a Clinical Trial Data Extraction Specialist operating in STRICT EXTRACTION MODE.

CRITICAL RULES:
1. You can ONLY extract information that appears in the provided document text
2. You have NO access to any clinical trial databases or prior knowledge
3. You CANNOT recall or use information from your training about any clinical trials
4. If information is not explicitly stated in the document, you MUST return NOT_FOUND
5. You must extract data VERBATIM - exactly as written in the document

HALLUCINATION PREVENTION:
- Never generate NCT numbers from memory
- Never recall drug names or dosages from training
- Never complete partial information with your knowledge
- Always return valid JSON when requested

You are extracting data for regulatory compliance where accuracy is critical. Incorrect extractions or hallucinations will result in serious consequences."""
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error querying GPT: {e}")
        return "Error in GPT query"


def target_specific_field(content: str, field_name: str, field_type: str, context: str = "") -> Any:
    """Extract a specific field with targeted prompting"""
    
    # Special handling for NCT number to prevent hallucination
    nct_warning = ""
    if field_name.lower() in ["nct number", "nct identifier", "clinicaltrials.gov identifier", "nct"]:
        nct_warning = """
    
    ⚠️ CRITICAL WARNING FOR NCT NUMBER EXTRACTION ⚠️
    - Extract ONLY the NCT number that appears in THIS document
    - Do NOT use NCT numbers from your memory or training data
    - The pattern is NCT followed by 8 digits
    - If you cannot find this pattern in the text below, return NOT_FOUND
    - Common mistake: Using NCT numbers from similar studies - DO NOT DO THIS
    """
    
    specific_prompt = f"""
    STRICT EXTRACTION MODE - DOCUMENT ONLY
    
    Extract the {field_name} from this clinical trial protocol text.

    {context}
    {nct_warning}

    EXTRACTION RULES:
    1. Search ONLY in the provided text below
    2. Return ONLY the exact {field_type} content of the {field_name}
    3. Extract VERBATIM as it appears in the document - do not modify or summarize
    4. If the information is not found in the text, respond with "NOT_FOUND"
    5. Do NOT use information from your training data or memory
    6. No additional text, explanations, or commentary

    TEXT TO SEARCH:
    {content}
    
    Remember: You can ONLY extract what you can see in the text above."""

    response = query_gpt(specific_prompt)

    if response == "NOT_FOUND" or response == "Error in GPT query":
        return None

    if field_type == "string":
        return response.strip()
    elif field_type == "array":
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            if '\n' in response:
                return [item.strip() for item in response.split('\n') if item.strip()]
            else:
                return [item.strip() for item in response.split(',') if item.strip()]
    elif field_type == "boolean":
        response = response.lower().strip()
        return response in ["yes", "true", "y", "1"]
    elif field_type == "full_text":
        return response
    else:
        return response


def extract_eligibility_criteria(content: str) -> Dict[str, Any]:
    """Special handling for eligibility criteria which is particularly challenging"""
    criteria_prompt = """
    Find and extract the EXACT and COMPLETE eligibility criteria section from this clinical trial protocol.
    Include ALL inclusion criteria and ALL exclusion criteria with EXACT text and formatting.
    Do not summarize or paraphrase. Extract the VERBATIM text as it appears.
    List all criteria (both inclusion and exclusion) exactly as shown in the document, including all bullet points and numbering.

    Return ONLY the exact criteria text with no additional explanations or commentary.
    """

    criteria_text = query_gpt(criteria_prompt + "\n\nDocument text:\n" + content)

    result = {
        "criteria": criteria_text if criteria_text != "Error in GPT query" and not criteria_text.startswith("I'm sorry") else "Not provided",
        "gender": "All",
        "minimum_age": "18 Years",
        "maximum_age": "N/A",
        "healthy_volunteers": "No"
    }

    if criteria_text == "Error in GPT query" or criteria_text.startswith("I'm sorry"):
        return result

    criteria_details_prompt = """
    Based on the eligibility criteria below, extract these specific details:

    Return ONLY a JSON object with these fields:
    - gender: The gender requirement ("All", "Female", or "Male")
    - minimum_age: The minimum age with units (e.g., "18 Years")
    - maximum_age: The maximum age with units or "N/A" if no limit
    - healthy_volunteers: Whether healthy volunteers are eligible ("Yes" or "No")

    Do not include any text outside the JSON object.

    Here are the criteria:
    """

    criteria_details = query_gpt(criteria_details_prompt + "\n" + criteria_text)

    try:
        json_start = criteria_details.find('{')
        if json_start > 0:
            criteria_details = criteria_details[json_start:]

        json_end = criteria_details.rfind('}')
        if json_end > 0 and len(criteria_details) > json_end + 1:
            criteria_details = criteria_details[:json_end+1]

        details = json.loads(criteria_details)

        if "gender" in details and details["gender"] in ["All", "Female", "Male"]:
            result["gender"] = details["gender"]

        if "minimum_age" in details:
            result["minimum_age"] = details["minimum_age"]

        if "maximum_age" in details:
            result["maximum_age"] = details["maximum_age"]

        if "healthy_volunteers" in details and details["healthy_volunteers"] in ["Yes", "No"]:
            result["healthy_volunteers"] = details["healthy_volunteers"]
    except json.JSONDecodeError:
        logger.warning("Failed to parse eligibility criteria details, using default values")

    return result


def extract_outcomes(content: str, outcome_type: str) -> List[Dict[str, str]]:
    """Extract primary or secondary outcomes with special handling"""
    outcomes_prompt = f"""
    Your task is to extract all {outcome_type} outcome measures from this clinical trial protocol.

    YOU MUST format your response as a valid JSON array, with each outcome as an object having these exact fields:
    - outcome_measure: string
    - outcome_time_frame: string
    - outcome_description: string

    If you cannot find any {outcome_type} outcomes, return an empty array: []

    DO NOT include any explanations, apologies, or text outside the JSON array.

    Here is the text:
    {content}
    """

    outcomes_result = query_gpt(outcomes_prompt)
    if outcomes_result == "Error in GPT query":
        return []

    try:
        json_start = outcomes_result.find('[')
        if json_start > 0:
            outcomes_result = outcomes_result[json_start:]

        json_end = outcomes_result.rfind(']')
        if json_end > 0 and len(outcomes_result) > json_end + 1:
            outcomes_result = outcomes_result[:json_end+1]

        outcomes = json.loads(outcomes_result)
        return outcomes
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse {outcome_type} outcomes JSON, trying alternative extraction")

        try:
            count_prompt = f"""
            How many distinct {outcome_type} outcome measures are explicitly defined in this clinical trial protocol?
            Return only a single number (e.g., "3"). If none are found, return "0".
            DO NOT include any other text, explanations, or apologies.
            """
            outcome_count = query_gpt(count_prompt).strip()

            outcome_count = ''.join(c for c in outcome_count if c.isdigit())

            if not outcome_count:
                logger.warning(f"Could not determine number of {outcome_type} outcomes")
                return []

            count = int(outcome_count)
            logger.info(f"Attempting to extract {count} {outcome_type} outcomes individually")

            outcomes = []
            for i in range(1, count + 1):
                measure_prompt = f"""
                What is the exact name or title of {outcome_type} outcome measure #{i} in this protocol?
                Return ONLY the name/title text with no additional explanation.
                """
                measure = query_gpt(measure_prompt).strip()

                time_frame_prompt = f"""
                What is the time frame specified for {outcome_type} outcome measure #{i} (titled: {measure}) in this protocol?
                Return ONLY the time frame with no additional explanation.
                """
                time_frame = query_gpt(time_frame_prompt).strip()

                description_prompt = f"""
                What is the full description of how {outcome_type} outcome measure #{i} (titled: {measure}) is assessed in this protocol?
                Return ONLY the description with no additional explanation.
                """
                description = query_gpt(description_prompt).strip()

                if measure and not measure.startswith("I'm sorry") and not measure.startswith("I don't"):
                    outcome = {
                        "outcome_measure": measure,
                        "outcome_time_frame": time_frame if not time_frame.startswith("I") else "Not specified",
                        "outcome_description": description if not description.startswith("I") else "Not specified"
                    }
                    outcomes.append(outcome)

            return outcomes
        except Exception as e:
            logger.warning(f"{outcome_type} outcomes extraction failed: {e}")
            return []


def extract_clinical_info(text_chunks: List[str]) -> Dict[str, Any]:
    """Extract structured clinical trial information from text chunks."""
    clinical_info = {}

    if not text_chunks:
        logger.error("No text chunks to process")
        text_chunks = ["Empty protocol document"]

    logger.info(f"Processing {len(text_chunks)} chunks of text")

    main_chunk = text_chunks[0] if len(text_chunks) > 0 else ""
    full_text = "\n\n".join(text_chunks[: min(3, len(text_chunks))])

    sections_prompt = """
    You are analyzing a clinical trial protocol document. Identify where the following sections are located:

    1. Title and basic info (beginning of document)
    2. Study design and phase information
    3. Eligibility criteria (inclusion/exclusion)
    4. Primary outcomes
    5. Secondary outcomes
    6. Arm groups/interventions
    7. Sponsor information

    For each section, provide the section name and whether it appears to be in the beginning, middle, or end of the document.
    Format as JSON with section_name and location fields.
    """

    section_info = query_gpt(sections_prompt + "\n\nHere's the beginning of the document:\n" + main_chunk)

    try:
        section_locations = json.loads(section_info)
    except Exception:
        logger.warning("Could not parse document structure, using default approach")
        section_locations = {}

    logger.info("Extracting basic study information")
    title_prompt = """
    Extract the EXACT official title and brief title of this clinical trial protocol.

    Return ONLY a JSON object with these fields:
    - brief_title: The short title of the study (usually shorter)
    - official_title: The full, complete title of the study (usually longer)
    - acronym: The study acronym or abbreviation if present (or null if none)

    Do not include any text outside the JSON object.
    """

    title_info = query_gpt(title_prompt + "\n\nDocument text:\n" + main_chunk)

    try:
        title_data = json.loads(title_info)
        clinical_info["brief_title"] = title_data.get("brief_title", "Unknown Title")
        clinical_info["official_title"] = title_data.get("official_title", "Unknown Official Title")
        if title_data.get("acronym"):
            clinical_info["acronym"] = title_data.get("acronym")
    except Exception:
        logger.warning("Failed to parse title information, falling back to field-by-field extraction")
        clinical_info["brief_title"] = target_specific_field(
            main_chunk,
            "brief title (short title) of the study",
            "string",
            "This is usually at the beginning of the protocol.",
        ) or "Unknown Title"
        clinical_info["official_title"] = target_specific_field(
            main_chunk,
            "official title (full title) of the study",
            "string",
            "This is usually at the beginning of the protocol and may be longer than the brief title.",
        ) or "Unknown Official Title"
        clinical_info["acronym"] = target_specific_field(
            main_chunk,
            "study acronym or abbreviation",
            "string",
            "This may appear near the title.",
        )

    clinical_info["org_study_id"] = target_specific_field(
        full_text,
        "organization's unique study identifier or protocol number",
        "string",
        "This is often a code or number that identifies the study within the organization.",
    ) or "UNKNOWN_ID"

    logger.info("Extracting study design information")
    design_prompt = """
    Extract the study type and phase information from this clinical trial protocol.

    Return ONLY a JSON object with these fields:
    - study_type: Either "Interventional", "Observational", or "Expanded Access"
    - phase: The study phase (e.g., "Phase 1", "Phase 2", "Phase 1/2", "N/A", etc.)
    - primary_purpose: For interventional studies, the primary purpose (e.g., "Treatment", "Prevention", etc.)

    Do not include any text outside the JSON object.
    """

    design_info = query_gpt(design_prompt + "\n\nDocument text:\n" + full_text)

    try:
        design_data = json.loads(design_info)
        study_type = design_data.get("study_type", "Interventional")
        phase = design_data.get("phase")
        primary_purpose = design_data.get("primary_purpose")
    except Exception:
        logger.warning("Failed to parse study design information, falling back to field-by-field extraction")
        study_type = target_specific_field(
            full_text,
            "study type",
            "string",
            "This should be one of: 'Interventional', 'Observational', or 'Expanded Access'.",
        ) or "Interventional"

        phase = target_specific_field(
            full_text,
            "study phase",
            "string",
            "For example: 'Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 1/2', etc.",
        )

        primary_purpose = target_specific_field(
            full_text,
            "primary purpose of the study",
            "string",
            "For interventional studies, this could be: Treatment, Prevention, Diagnostic, etc.",
        )

    clinical_info["study_design"] = {
        "study_type": study_type,
    }

    if study_type == "Interventional" or not study_type:
        interventional_fields: Dict[str, Any] = {}
        interventional_fields["interventional_subtype"] = primary_purpose or "Treatment"
        interventional_fields["phase"] = phase or "N/A"

        assignment = target_specific_field(
            full_text,
            "study design/interventional study model",
            "string",
            "This could be: Single Group Assignment, Parallel Assignment, Crossover Assignment, etc.",
        )
        if assignment:
            interventional_fields["assignment"] = assignment

        allocation = target_specific_field(
            full_text,
            "allocation method",
            "string",
            "This should be one of: 'Randomized', 'Non-randomized', or 'N/A'.",
        )
        interventional_fields["allocation"] = allocation or "N/A"

        masking_prompt = """
        Extract the masking/blinding information from this clinical trial protocol.

        Return ONLY a JSON object with these fields:
        - no_masking: "yes" if the study has no masking/blinding, "no" if it has some form of masking
        - masked_subject: "yes" if subjects are masked/blinded, "no" if not (if applicable)
        - masked_caregiver: "yes" if caregivers are masked/blinded, "no" if not (if applicable)
        - masked_investigator: "yes" if investigators are masked/blinded, "no" if not (if applicable)
        - masked_assessor: "yes" if outcome assessors are masked/blinded, "no" if not (if applicable)
        - description: A description of how masking was performed (if applicable)

        Only include fields that are relevant and can be determined from the protocol.
        """

        masking_result = query_gpt(masking_prompt + "\n\nDocument text:\n" + full_text)

        try:
            masking_info = json.loads(masking_result)
            if "no_masking" in masking_info:
                interventional_fields.setdefault("masking", {})["no_masking"] = masking_info["no_masking"]
            for role in ["subject", "caregiver", "investigator", "assessor"]:
                key = f"masked_{role}"
                if key in masking_info:
                    interventional_fields.setdefault("masking", {})[key] = masking_info[key]
            if "description" in masking_info:
                interventional_fields.setdefault("masking", {})["description"] = masking_info["description"]
        except json.JSONDecodeError:
            logger.warning("Failed to parse masking information")

        clinical_info["study_design"]["interventional_design"] = interventional_fields

    elif study_type == "Observational":
        obs_prompt = """
        Extract the observational study design information from this clinical trial protocol.

        Return ONLY a JSON object with these fields:
        - observational_study_design: Type of observational study (e.g., Cohort, Case-Control, etc.)
        - timing: Timing of observational model (e.g., Prospective, Retrospective, Cross-Sectional)
        - biospecimen_retention: Type of biospecimen retention (e.g., One Time, Sample w/ DNA, etc.)
        - biospecimen_description: Description of what biospecimens are retained (if applicable)
        - number_of_groups: Number of groups/cohorts in the study
        - patient_registry: "yes" if this is a patient registry, "no" otherwise
        - target_duration_quantity: Quantity for target duration if applicable
        - target_duration_units: Units for target duration if applicable

        Only include fields that are relevant.

        Do not include any text outside the JSON object.
        """

        obs_result = query_gpt(obs_prompt + "\n\nDocument text:\n" + full_text)

        try:
            obs_data = json.loads(obs_result)
            clinical_info["study_design"]["observational_design"] = obs_data
        except json.JSONDecodeError:
            logger.warning("Failed to parse observational design information")

    logger.info("Extracting eligibility criteria")
    criteria = extract_eligibility_criteria(full_text)
    clinical_info["eligibility"] = criteria

    logger.info("Extracting primary outcomes")
    primary_outcomes = extract_outcomes(full_text, "primary")
    if primary_outcomes:
        clinical_info["primary_outcomes"] = primary_outcomes

    logger.info("Extracting secondary outcomes")
    secondary_outcomes = extract_outcomes(full_text, "secondary")
    if secondary_outcomes:
        clinical_info["secondary_outcomes"] = secondary_outcomes

    arm_chunk = "\n\n".join(text_chunks)
    logger.info("Extracting arm group information")
    arm_groups_prompt = """
    Extract the arm group information from this clinical trial protocol.

    Return ONLY a JSON array where each object has these fields:
    - arm_group_label: The name of the arm/group
    - arm_type: The type of the arm (e.g., Experimental, Placebo Comparator, etc.)
    - arm_group_description: Description of the arm

    If you cannot find any arm groups, return an empty array: []

    Do not include any text outside the JSON array.
    """

    arm_groups_result = query_gpt(arm_groups_prompt + "\n\nDocument text:\n" + arm_chunk)

    try:
        json_start = arm_groups_result.find("[")
        if json_start > 0:
            arm_groups_result = arm_groups_result[json_start:]

        json_end = arm_groups_result.rfind("]")
        if json_end > 0 and len(arm_groups_result) > json_end + 1:
            arm_groups_result = arm_groups_result[: json_end + 1]

        arm_groups = json.loads(arm_groups_result)
        if arm_groups:
            clinical_info["arm_groups"] = arm_groups
    except json.JSONDecodeError:
        logger.warning("Failed to parse arm groups JSON")

    logger.info("Extracting interventions")
    interventions_prompt = """
    Extract all interventions from this clinical trial protocol.

    Return ONLY a JSON array where each object has these fields:
    - intervention_type: The type (e.g., "Drug", "Device", "Biological/Vaccine", etc.)
    - intervention_name: The name of the intervention
    - intervention_description: A description of the intervention
    - arm_group_label: Array of strings with names of arms that receive this intervention
    - intervention_other_name: Array of strings with alternative names (or empty array)

    If you cannot find intervention information, return an empty array: []

    Do not include any text outside the JSON array.
    """

    interventions_result = query_gpt(interventions_prompt + "\n\nDocument text:\n" + (arm_chunk or full_text))

    try:
        json_start = interventions_result.find("[")
        if json_start > 0:
            interventions_result = interventions_result[json_start:]

        json_end = interventions_result.rfind("]")
        if json_end > 0 and len(interventions_result) > json_end + 1:
            interventions_result = interventions_result[: json_end + 1]

        interventions = json.loads(interventions_result)
        if interventions:
            clinical_info["interventions"] = interventions
    except json.JSONDecodeError:
        logger.warning("Failed to parse interventions JSON")

    logger.info("Extracting sponsor information")
    sponsors_prompt = """
    Extract the sponsor information from this clinical trial protocol.

    Return ONLY a JSON object with these fields:
    - lead_sponsor: The organization name of the primary sponsor
    - collaborators: Array of organization names of any collaborators
    - responsible_party_type: Type (e.g., "Sponsor", "Principal Investigator", "Sponsor-Investigator")
    - investigator_title: Title of the investigator (if applicable)
    - investigator_affiliation: Affiliation of the investigator (if applicable)

    Only include fields that are present in the document.

    Do not include any text outside the JSON object.
    """

    sponsors_result = query_gpt(sponsors_prompt + "\n\nDocument text:\n" + main_chunk)

    try:
        sponsors_data = json.loads(sponsors_result)

        sponsors = {"lead_sponsor": sponsors_data.get("lead_sponsor", "UNKNOWN_SPONSOR")}
        if "collaborators" in sponsors_data:
            sponsors["collaborators"] = sponsors_data["collaborators"]

        if any(key in sponsors_data for key in ["responsible_party_type", "investigator_title", "investigator_affiliation"]):
            resp_party = {}
            if "responsible_party_type" in sponsors_data:
                resp_party["resp_party_type"] = sponsors_data["responsible_party_type"]
            if "investigator_title" in sponsors_data:
                resp_party["investigator_title"] = sponsors_data["investigator_title"]
            if "investigator_affiliation" in sponsors_data:
                resp_party["investigator_affiliation"] = sponsors_data["investigator_affiliation"]
            sponsors["responsible_party"] = resp_party

        clinical_info["sponsors"] = sponsors
    except json.JSONDecodeError:
        logger.warning("Failed to parse sponsors JSON")
        lead_sponsor = target_specific_field(
            main_chunk,
            "lead sponsor",
            "string",
            "What organization is the primary/lead sponsor of this study?",
        )
        if lead_sponsor:
            clinical_info["sponsors"] = {"lead_sponsor": lead_sponsor}

    logger.info("Extracting study details (enrollment, status, dates, conditions)")
    details_prompt = """
    Extract these key study details from the clinical trial protocol.

    Return ONLY a JSON object with these fields:
    - enrollment: The target enrollment number (numeric value as string)
    - enrollment_type: "Anticipated" or "Actual"
    - overall_status: Study status (e.g., "Not yet recruiting", "Recruiting", "Completed")
    - start_date: Start date in YYYY-MM format
    - start_date_type: "Anticipated" or "Actual"
    - primary_compl_date: Primary completion date in YYYY-MM format
    - primary_compl_date_type: "Anticipated" or "Actual"
    - conditions: Array of strings with medical conditions being studied
    - keywords: Array of strings with relevant keywords

    Only include fields that you can find in the document. If information isn't available, omit the field.

    Do not include any text outside the JSON object.
    """

    details_result = query_gpt(details_prompt + "\n\nDocument text:\n" + full_text)

    try:
        details_data = json.loads(details_result)
        for field in [
            "enrollment",
            "enrollment_type",
            "overall_status",
            "start_date",
            "start_date_type",
            "primary_compl_date",
            "primary_compl_date_type",
            "conditions",
            "keywords",
        ]:
            if field in details_data:
                clinical_info[field] = details_data[field]
    except json.JSONDecodeError:
        logger.warning("Failed to parse study details")

    logger.info("Extracting study summary information")
    summary_prompt = """
    Extract the brief summary and detailed description of this clinical trial.

    Return ONLY a JSON object with these fields:
    - brief_summary: A brief summary of the study's purpose and approach (1-3 sentences)
    - detailed_description: A more detailed description of the study (if available)

    Only include fields that you can find in the document.

    Do not include any text outside the JSON object.
    """

    summary_result = query_gpt(summary_prompt + "\n\nDocument text:\n" + full_text)

    try:
        summary_data = json.loads(summary_result)
        if "brief_summary" in summary_data:
            clinical_info["brief_summary"] = summary_data["brief_summary"]
        if "detailed_description" in summary_data:
            clinical_info["detailed_description"] = summary_data["detailed_description"]
    except json.JSONDecodeError:
        logger.warning("Failed to parse summary information")

    return clinical_info

