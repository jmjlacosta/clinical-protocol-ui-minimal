import os
import argparse
import logging
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from typing import Dict, Any, List

from .ingestion import extract_text_from_pdf, chunk_text
from .prompts import (
    query_gpt,
    target_specific_field,
    extract_eligibility_criteria,
    extract_outcomes,
    extract_clinical_info,
)
from .xml_builder import generate_xml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

__all__ = [
    'extract_text_from_pdf',
    'chunk_text',
    'query_gpt',
    'target_specific_field',
    'extract_eligibility_criteria',
    'extract_outcomes',
    'extract_clinical_info',
    'generate_xml',
    'process_pdf_to_xml',
]


def process_pdf_to_xml(pdf_path: str, output_xml_path: str | None = None) -> str:
    """Process a PDF file to XML conforming to ClinicalTrials.gov schema"""
    try:
        logger.info(f"Extracting text from PDF: {pdf_path}")
        pdf_text = extract_text_from_pdf(pdf_path)

        logger.info("Processing text")
        text_chunks = chunk_text(pdf_text)

        logger.info(f"Extracting clinical information from {len(text_chunks)} chunks of text")
        clinical_info = extract_clinical_info(text_chunks)

        logger.info("Generating XML")
        xml_content = generate_xml(clinical_info)

        if output_xml_path:
            with open(output_xml_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)
            logger.info(f"XML saved to: {output_xml_path}")

        return xml_content

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)

        root = ET.Element("study_collection", xmlns="http://clinicaltrials.gov/prs")
        study = ET.SubElement(root, "clinical_study")

        id_info = ET.SubElement(study, "id_info")
        org_name = ET.SubElement(id_info, "org_name")
        org_name.text = "ERROR_PROCESSING"

        org_study_id = ET.SubElement(id_info, "org_study_id")
        org_study_id.text = os.path.basename(pdf_path).replace(".pdf", "")

        brief_title = ET.SubElement(study, "brief_title")
        brief_title.text = f"Error processing {os.path.basename(pdf_path)}"

        error_desc = ET.SubElement(study, "detailed_description")
        error_textblock = ET.SubElement(error_desc, "textblock")
        error_textblock.text = f"Error during processing: {str(e)}"

        rough_string = ET.tostring(root, encoding='utf-8')
        reparsed = minidom.parseString(rough_string)
        xml_content = reparsed.toprettyxml(indent="  ")

        if output_xml_path:
            with open(output_xml_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)
            logger.info(f"Minimal XML saved to: {output_xml_path} due to error: {e}")

        return xml_content


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert Clinical Trial Protocol PDF to XML')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--output', '-o', help='Output XML file path')

    args = parser.parse_args()

    output_path = args.output
    if not output_path:
        pdf_base = os.path.splitext(os.path.basename(args.pdf_path))[0]
        output_path = f"{pdf_base}_protocol.xml"

    try:
        xml_content = process_pdf_to_xml(args.pdf_path, output_path)
        print(f"Successfully converted {args.pdf_path} to {output_path}")
    except Exception as e:
        logger.error(f"Unhandled error in main: {e}")
        print(f"Error processing PDF: {e}")


if __name__ == '__main__':

    main()
