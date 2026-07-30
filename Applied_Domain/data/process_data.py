import io
import os
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# Checking for PyPDF availability for PDF parsing
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

# Path configuration
DATA_DIR = Path(__file__).resolve().parent / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1. HTML Web Sources
WEB_SOURCES = {
    "cdc-heart-disease-prevention": "https://www.cdc.gov/heart-disease/prevention/index.html",
    "nhs-common-cold": "https://www.nhs.uk/conditions/common-cold/",
    "nhs-heartburn-and-acid-reflux": "https://www.nhs.uk/conditions/heartburn-and-acid-reflux/",
    "nhs-constipation": "https://www.nhs.uk/conditions/constipation/",
    "nhs-diarrhoea": "https://www.nhs.uk/conditions/diarrhoea/",
    "nhs-fever-in-adults": "https://www.nhs.uk/conditions/fever-in-adults/",
    "nhs-sprains-and-strains": "https://www.nhs.uk/conditions/sprains-and-strains/",
    "nhs-urinary-tract-infections-utis": "https://www.nhs.uk/conditions/urinary-tract-infections-utis/",
    "nhs-hypertension-guidelines": "https://www.nhs.uk/conditions/high-blood-pressure-hypertension/",
    "nhs-cardiovascular-disease": "https://www.nhs.uk/conditions/cardiovascular-disease/",
    "nhs-cpr-first-aid": "https://www.nhs.uk/conditions/first-aid/cpr/"
}

# 2. PDF Guidelines Sources
PDF_SOURCES = {
    "who-hypertension-treatment": "https://iris.who.int/server/api/core/bitstreams/f062769d-f075-4a00-87af-0a2106e0bd04/content",
    "who-basic-emergency-care": "https://hlh.who.int/docs/librariesprovider4/hlh-documents/who-icrc-basic-emergency-care.pdf",
    "who-mhGap": "https://iris.who.int/server/api/core/bitstreams/93e56376-0e64-4a08-bdd0-703313be5354/content",
    "who-communicable-disease-sheet": "https://www.who.int/docs/default-source/documents/publications/information-sheet-communicable-diseases.pdf?sfvrsn=384d78b2_1",
    "icmr-nhm-malaria-treatment": "https://www.nhm.gov.in/images/pdf/guidelines/nrhm-guidelines/stg/malaria-stg.pdf",
    "cdc-dengue-clinical-management": "https://www.cdc.gov/dengue/resources/dengue-clinician-guide_508.pdf" 
}


def process_web_pages():
    """Scrapes HTML pages, removes navigation noise, and converts body content to Markdown."""
    print("\n--- Scraping Web Guidelines ---")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    for name, url in WEB_SOURCES.items():
        out_file = DATA_DIR / f"{name}.md"
        if out_file.exists():
            print(f"⏩ Skipping {name} (already downloaded)")
            continue

        try:
            print(f"Fetching {name}...")
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Removing unwanted HTML elements
            for tag in soup(["nav", "header", "footer", "script", "style", "aside"]):
                tag.decompose()

            # Target main content area if present, otherwise go to body
            content = soup.find("main") or soup.find("article") or soup.body
            markdown_content = md(str(content), heading_style="ATX")

            with open(out_file, "w", encoding="utf-8") as f:
                f.write(f"# {name.replace('-', ' ').title()}\nSource: {url}\n\n")
                f.write(markdown_content)

            print(f"✅ Saved web guideline: {out_file.name}")

        except Exception as e:
            print(f"❌ Error processing {name}: {e}")


def process_pdf_documents():
    """Downloads medical PDFs and extracts text into Markdown files."""
    print("\n--- Downloading & Extracting PDF Guidelines ---")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    for name, url in PDF_SOURCES.items():
        out_file = DATA_DIR / f"{name}.md"
        if out_file.exists():
            print(f"⏩ Skipping {name} (already downloaded)")
            continue

        try:
            print(f"Downloading PDF {name}...")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            if HAS_PYPDF:
                pdf_file = io.BytesIO(response.content)
                reader = PdfReader(pdf_file)

                extracted_text = [f"# {name.replace('-', ' ').title()}\nSource: {url}\n\n"]
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        extracted_text.append(f"## Page {page_num}\n\n{text.strip()}\n")

                full_markdown = "\n".join(extracted_text)

                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(full_markdown)

                print(f"✅ Extracted PDF text to: {out_file.name}")
            else:
                # Save as raw PDF if pypdf is missing
                pdf_raw = DATA_DIR / f"{name}.pdf"
                with open(pdf_raw, "wb") as f:
                    f.write(response.content)
                print(f"⚠️ PyPDF not installed. Saved raw PDF as: {pdf_raw.name}")

        except Exception as e:
            print(f"❌ Error downloading {name}: {e}")


if __name__ == "__main__":
    print(f"Target Directory: {DATA_DIR.resolve()}")
    process_web_pages()
    process_pdf_documents()
    print("\n🎉 Data collection process complete!")