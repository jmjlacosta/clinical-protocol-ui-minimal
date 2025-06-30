import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_xml(clinical_info: Dict[str, Any]) -> str:
    """Generate XML conforming to the ClinicalTrials.gov schema"""
    root = ET.Element("study_collection", xmlns="http://clinicaltrials.gov/prs")
    study = ET.SubElement(root, "clinical_study")

    id_info = ET.SubElement(study, "id_info")

    org_name = ET.SubElement(id_info, "org_name")
    org_name.text = clinical_info.get("org_name", "UNKNOWN_ORG")

    org_study_id = ET.SubElement(id_info, "org_study_id")
    org_study_id.text = clinical_info.get("org_study_id", "UNKNOWN_ID")

    if "brief_title" in clinical_info:
        brief_title = ET.SubElement(study, "brief_title")
        brief_title.text = clinical_info["brief_title"]

    if "official_title" in clinical_info:
        official_title = ET.SubElement(study, "official_title")
        official_title.text = clinical_info["official_title"]

    if "acronym" in clinical_info:
        acronym = ET.SubElement(study, "acronym")
        acronym.text = clinical_info["acronym"]

    if "sponsors" in clinical_info:
        sponsors = ET.SubElement(study, "sponsors")
        lead_sponsor = ET.SubElement(sponsors, "lead_sponsor")
        agency = ET.SubElement(lead_sponsor, "agency")
        agency.text = clinical_info["sponsors"].get("lead_sponsor", "UNKNOWN_SPONSOR")

        if "collaborators" in clinical_info["sponsors"]:
            for collab in clinical_info["sponsors"]["collaborators"]:
                collaborator = ET.SubElement(sponsors, "collaborator")
                collab_agency = ET.SubElement(collaborator, "agency")
                collab_agency.text = collab

        if "responsible_party" in clinical_info["sponsors"]:
            resp_party = ET.SubElement(sponsors, "resp_party")
            resp_party_type = ET.SubElement(resp_party, "resp_party_type")
            resp_party_type.text = clinical_info["sponsors"]["responsible_party"].get("resp_party_type", "Sponsor")

            if "investigator_title" in clinical_info["sponsors"]["responsible_party"]:
                inv_title = ET.SubElement(resp_party, "investigator_title")
                inv_title.text = clinical_info["sponsors"]["responsible_party"]["investigator_title"]

            if "investigator_affiliation" in clinical_info["sponsors"]["responsible_party"]:
                inv_affil = ET.SubElement(resp_party, "investigator_affiliation")
                inv_affil.text = clinical_info["sponsors"]["responsible_party"]["investigator_affiliation"]

    study_design = ET.SubElement(study, "study_design")

    study_type = ET.SubElement(study_design, "study_type")
    study_type.text = clinical_info.get("study_design", {}).get("study_type", "Interventional")

    if study_type.text == "Interventional" and "interventional_design" in clinical_info.get("study_design", {}):
        int_design = ET.SubElement(study_design, "interventional_design")
        int_design_data = clinical_info["study_design"]["interventional_design"]

        subtype = ET.SubElement(int_design, "interventional_subtype")
        subtype.text = int_design_data.get("interventional_subtype", "Treatment")

        phase = ET.SubElement(int_design, "phase")
        phase.text = int_design_data.get("phase", "N/A")

        if "assignment" in int_design_data:
            assignment = ET.SubElement(int_design, "assignment")
            assignment.text = int_design_data["assignment"]

        allocation = ET.SubElement(int_design, "allocation")
        allocation.text = int_design_data.get("allocation", "N/A")

        if "masking" in int_design_data:
            masking_data = int_design_data["masking"]

            if "no_masking" in masking_data:
                no_masking = ET.SubElement(int_design, "no_masking")
                no_masking.text = masking_data["no_masking"]

            for role in ["subject", "caregiver", "investigator", "assessor"]:
                masked_key = f"masked_{role}"
                if masked_key in masking_data:
                    masked_elem = ET.SubElement(int_design, masked_key)
                    masked_elem.text = masking_data[masked_key]

            if "description" in masking_data:
                masking_desc = ET.SubElement(int_design, "masking_description")
                textblock = ET.SubElement(masking_desc, "textblock")
                textblock.text = masking_data["description"]

    elif study_type.text == "Observational" and "observational_design" in clinical_info.get("study_design", {}):
        obs_design = ET.SubElement(study_design, "observational_design")
        obs_design_data = clinical_info["study_design"]["observational_design"]

        obs_study_design = ET.SubElement(obs_design, "observational_study_design")
        obs_study_design.text = obs_design_data.get("observational_study_design", "Other")

        timing = ET.SubElement(obs_design, "timing")
        timing.text = obs_design_data.get("timing", "Other")

        biospec_retention = ET.SubElement(obs_design, "biospecimen_retention")
        biospec_retention.text = obs_design_data.get("biospecimen_retention", "None Retained")

        if "biospecimen_description" in obs_design_data:
            biospec_desc = ET.SubElement(obs_design, "biospecimen_description")
            textblock = ET.SubElement(biospec_desc, "textblock")
            textblock.text = obs_design_data["biospecimen_description"]

        num_groups = ET.SubElement(obs_design, "number_of_groups")
        num_groups.text = obs_design_data.get("number_of_groups", "1")

        if "patient_registry" in obs_design_data:
            patient_reg = ET.SubElement(obs_design, "patient_registry")
            patient_reg.text = obs_design_data["patient_registry"]

            if obs_design_data["patient_registry"] == "yes":
                if "target_duration_quantity" in obs_design_data:
                    target_dur_qty = ET.SubElement(obs_design, "target_duration_quantity")
                    target_dur_qty.text = obs_design_data["target_duration_quantity"]

                if "target_duration_units" in obs_design_data:
                    target_dur_units = ET.SubElement(obs_design, "target_duration_units")
                    target_dur_units.text = obs_design_data["target_duration_units"]

    if "eligibility" in clinical_info:
        eligibility_data = clinical_info["eligibility"]
        eligibility = ET.SubElement(study, "eligibility")

        criteria = ET.SubElement(eligibility, "criteria")
        textblock = ET.SubElement(criteria, "textblock")
        textblock.text = eligibility_data.get("criteria", "Not provided")

        gender = ET.SubElement(eligibility, "gender")
        gender.text = eligibility_data.get("gender", "All")

        minimum_age = ET.SubElement(eligibility, "minimum_age")
        minimum_age.text = eligibility_data.get("minimum_age", "18 Years")

        maximum_age = ET.SubElement(eligibility, "maximum_age")
        maximum_age.text = eligibility_data.get("maximum_age", "N/A")

        healthy_volunteers = ET.SubElement(eligibility, "healthy_volunteers")
        healthy_volunteers.text = eligibility_data.get("healthy_volunteers", "No")

    if "primary_outcomes" in clinical_info:
        for outcome in clinical_info["primary_outcomes"]:
            primary_outcome = ET.SubElement(study, "primary_outcome")
            outcome_measure = ET.SubElement(primary_outcome, "outcome_measure")
            outcome_measure.text = outcome.get("outcome_measure", "")

            if "outcome_time_frame" in outcome:
                time_frame = ET.SubElement(primary_outcome, "outcome_time_frame")
                time_frame.text = outcome["outcome_time_frame"]

            if "outcome_description" in outcome:
                desc = ET.SubElement(primary_outcome, "outcome_description")
                textblock = ET.SubElement(desc, "textblock")
                textblock.text = outcome["outcome_description"]

    if "secondary_outcomes" in clinical_info:
        for outcome in clinical_info["secondary_outcomes"]:
            secondary_outcome = ET.SubElement(study, "secondary_outcome")
            outcome_measure = ET.SubElement(secondary_outcome, "outcome_measure")
            outcome_measure.text = outcome.get("outcome_measure", "")

            if "outcome_time_frame" in outcome:
                time_frame = ET.SubElement(secondary_outcome, "outcome_time_frame")
                time_frame.text = outcome["outcome_time_frame"]

            if "outcome_description" in outcome:
                desc = ET.SubElement(secondary_outcome, "outcome_description")
                textblock = ET.SubElement(desc, "textblock")
                textblock.text = outcome["outcome_description"]

    if "enrollment" in clinical_info:
        enrollment = ET.SubElement(study, "enrollment")
        enrollment.text = clinical_info["enrollment"]

        if "enrollment_type" in clinical_info:
            enrollment_type = ET.SubElement(study, "enrollment_type")
            enrollment_type.text = clinical_info["enrollment_type"]

    if "conditions" in clinical_info:
        for condition in clinical_info["conditions"]:
            condition_elem = ET.SubElement(study, "condition")
            condition_elem.text = condition

    if "keywords" in clinical_info:
        for keyword in clinical_info["keywords"]:
            keyword_elem = ET.SubElement(study, "keyword")
            keyword_elem.text = keyword

    if "arm_groups" in clinical_info:
        for arm in clinical_info["arm_groups"]:
            arm_group = ET.SubElement(study, "arm_group")

            arm_label = ET.SubElement(arm_group, "arm_group_label")
            arm_label.text = arm.get("arm_group_label", "")

            if "arm_type" in arm:
                arm_type = ET.SubElement(arm_group, "arm_type")
                arm_type.text = arm["arm_type"]

            if "arm_group_description" in arm:
                arm_desc = ET.SubElement(arm_group, "arm_group_description")
                textblock = ET.SubElement(arm_desc, "textblock")
                textblock.text = arm["arm_group_description"]

    if "interventions" in clinical_info:
        for intervention in clinical_info["interventions"]:
            intervention_elem = ET.SubElement(study, "intervention")

            if "intervention_type" in intervention:
                int_type = ET.SubElement(intervention_elem, "intervention_type")
                int_type.text = intervention["intervention_type"]

            int_name = ET.SubElement(intervention_elem, "intervention_name")
            int_name.text = intervention.get("intervention_name", "")

            if "intervention_description" in intervention:
                int_desc = ET.SubElement(intervention_elem, "intervention_description")
                textblock = ET.SubElement(int_desc, "textblock")
                textblock.text = intervention["intervention_description"]

            if "arm_group_label" in intervention:
                for label in intervention["arm_group_label"]:
                    arm_label = ET.SubElement(intervention_elem, "arm_group_label")
                    arm_label.text = label

            if "intervention_other_name" in intervention:
                for other_name in intervention["intervention_other_name"]:
                    other_name_elem = ET.SubElement(intervention_elem, "intervention_other_name")
                    other_name_elem.text = other_name

    for field in [
        "overall_status",
        "start_date",
        "start_date_type",
        "primary_compl_date",
        "primary_compl_date_type",
        "last_follow_up_date",
        "last_follow_up_date_type",
    ]:
        if field in clinical_info:
            field_elem = ET.SubElement(study, field)
            field_elem.text = clinical_info[field]

    if "brief_summary" in clinical_info:
        brief_summary = ET.SubElement(study, "brief_summary")
        textblock = ET.SubElement(brief_summary, "textblock")
        textblock.text = clinical_info["brief_summary"]

    if "detailed_description" in clinical_info:
        detailed_desc = ET.SubElement(study, "detailed_description")
        textblock = ET.SubElement(detailed_desc, "textblock")
        textblock.text = clinical_info["detailed_description"]

    rough_string = ET.tostring(root, encoding="utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")
