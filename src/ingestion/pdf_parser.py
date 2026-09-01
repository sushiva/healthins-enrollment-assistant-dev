"""Table-aware PDF extraction for formulary documents.

Ported from notebooks/formulary/02-Basic_RAG_Formulary.ipynb §2 (originally
from langchain_03_table_aware.ipynb). Keeps each drug's full record —
section, name, tier, requirements — as one self-contained Document, so a
drug's fields can never be split across chunks the way a fixed-size
character-window chunker would split a formulary table.

Hardened per codereview/pdf_parser_review.md: case-insensitive header
detection, a None-safe extract_tables() call, a "source" field in metadata,
and str|Path input with an explicit missing-file error.
"""
from pathlib import Path

import pdfplumber
from langchain_core.documents import Document


def extract_drug_rows(path: str | Path) -> list[Document]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"PDF file not found at: {file_path}")

    docs = []
    current_section = None  # persists across pages until the next section header

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables() or []  # None on pages with no detected table
            for table in tables:
                for row in table:
                    cells = [(c or "").strip().replace("\n", " ") for c in row]
                    non_empty = [c for c in cells if c]

                    if not non_empty:
                        continue
                    if any("drug name" in c.lower() for c in non_empty):  # repeated table header row
                        continue
                    if len(non_empty) == 1:  # section header, e.g. "OPIOID ANALGESICS"
                        current_section = non_empty[0]
                        continue
                    if len(cells) < 4:  # page footer / legend row
                        continue

                    _, drug_name, tier, requirements = cells[:4]
                    if not drug_name:
                        continue

                    content = (
                        f"Section: {current_section or 'N/A'}\n"
                        f"Drug: {drug_name}\n"
                        f"Tier: {tier or 'N/A'}\n"
                        f"Requirements/Limits: {requirements or 'None'}"
                    )
                    docs.append(Document(
                        page_content=content,
                        metadata={
                            "source": str(file_path),
                            "section": current_section or "N/A",
                            "page": page_num,
                            "drug": drug_name,
                        },
                    ))
    return docs
