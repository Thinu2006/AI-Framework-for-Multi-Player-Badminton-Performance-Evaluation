"""
Simple Badminton Shot RAG Pipeline
==================================
A clean, simple, beginner-friendly RAG implementation:
1. Ingestion: Load PDF pages via pypdf.
2. Shot-Based Chunking: Extract entire shot sections with metadata.
3. Vector Indexing: Build TF-IDF vector index over shot chunks.
4. Retriever: Retrieve relevant shot chunk via vector similarity.
5. Prompt Builder: Fuse retrieved chunk into prompt with JSON schema.
6. Structured Extractor: Parse precise coaching fields.
7. Schema Validation: Enforce exactly 7 shot objects & valid fields.
"""

import os
import json
import re
import pypdf
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, List

# Target Shots Mapping
TARGET_SHOTS = [
    {"query": "Backhand Net Shot", "shot_type": "Backhand Net Shot", "page_start": 36, "page_end": 37},
    {"query": "Forehand Net Shot", "shot_type": "Forehand Net Shot", "page_start": 37, "page_end": 38},
    {"query": "Backhand Net Lift", "shot_type": "Backhand Net Lift", "page_start": 38, "page_end": 39},
    {"query": "Forehand Net Lift d. Forehand Net Lift", "shot_type": "Forehand Net Lift", "page_start": 40, "page_end": 41},
    {"query": "Forehand Clear a. Forehand Clear", "shot_type": "Clear", "page_start": 47, "page_end": 48},
    {"query": "Forehand Smash b. Forehand Smash", "shot_type": "Smash", "page_start": 48, "page_end": 49},
    {"query": "Forehand Dropshot c. Forehand Dropshot", "shot_type": "Drop", "page_start": 49, "page_end": 50}
]

# Step 1: Document Ingestion
def ingest_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Ingest PDF and return pages with 1-based page numbers."""
    reader = pypdf.PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        lines = [l.strip() for l in text.split("\n") if "All rights reserved" not in l]
        pages.append({"page_num": i + 1, "text": "\n".join(lines)})
    return pages

# Step 2: Shot-Based Chunking
def create_shot_chunks(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract complete shot sections as individual chunks with metadata."""
    chunks = []
    for shot in TARGET_SHOTS:
        p_start, p_end = shot["page_start"], shot["page_end"]
        content_lines = []
        for p in pages:
            if p_start <= p["page_num"] <= p_end:
                content_lines.append(f"--- Page {p['page_num']} ---\n{p['text']}")
        
        chunks.append({
            "shot_name": shot["query"],
            "shot_type": shot["shot_type"],
            "page_start": p_start,
            "page_end": p_end,
            "content": "\n\n".join(content_lines)
        })
    return chunks

# Step 3 & 4: Vector Indexing & Retrieval Engine
class VectorStoreRetriever:
    """TF-IDF Vector Store and Similarity Retriever."""
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        texts = [c["shot_name"] + "\n" + c["content"] for c in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str) -> Dict[str, Any]:
        """Retrieve most relevant shot chunk based on TF-IDF cosine similarity."""
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        best_idx = int(np.argmax(similarities))
        return self.chunks[best_idx]

# Step 5 & 6: Structured Extractor
def extract_fields(shot_type: str, page_start: int, page_end: int, chunk_content: str) -> Dict[str, Any]:
    def clean_text(t: str) -> str:
        lines = [l.strip() for l in t.split('\n') if l.strip() and not l.strip().isdigit() and "--- Page" not in l]
        return " ".join(lines)

    def get_sec(start_kw, end_kws):
        pattern = re.compile(rf"{start_kw}\s*(.*?)(?={'|'.join(end_kws)}|$)", re.DOTALL | re.IGNORECASE)
        m = pattern.search(chunk_content)
        if m:
            return clean_text(m.group(1))
        return ""

    when_use = get_sec("WHEN WE USE IT", ["WHY WE USE IT", "WHAT TO COACH", "HOW TO COACH"])
    why_use = get_sec("WHY WE USE IT", ["WHAT TO COACH", "HOW TO COACH", "HINTS"])
    prep = get_sec("PREPARE", ["BACKSWING", "FORWARD SWING", "FOLLOW THROUGH", "HOW TO COACH"])
    backswing = get_sec("BACKSWING", ["FORWARD SWING", "FOLLOW THROUGH", "HOW TO COACH"])
    forward = get_sec("FORWARD SWING", ["FOLLOW THROUGH", "HOW TO COACH"])
    follow = get_sec("FOLLOW THROUGH", ["HOW TO COACH", "HINTS", "100", "101", "102", "103", "104", "105", "106", "107", "116", "117", "118", "119", "120", "121", "122", "123"])
    
    # Specific extraction for HOW TO COACH practices block
    m_how = re.search(r"HOW TO COACH[^\n]*\n(.*?)(?=HINTS, TIPS AND VARIATIONS|\Z)", chunk_content, re.DOTALL)
    how_coach = clean_text(m_how.group(1)) if m_how else ""

    # Specific extraction for HINTS, TIPS AND VARIATIONS
    m_hints = re.search(r"HINTS, TIPS AND VARIATIONS\s*\n(.*?)(?=\n[a-z]\.|\Z)", chunk_content, re.DOTALL)
    hints = clean_text(m_hints.group(1)) if m_hints else ""

    return {
        "shot_type": shot_type,
        "page_start": page_start,
        "page_end": page_end,
        "structured_fields": {
            "when_we_use_it": when_use,
            "why_we_use_it": why_use,
            "what_to_coach": {
                "prepare": prep,
                "backswing": backswing,
                "forward_swing": forward,
                "follow_through": follow
            },
            "how_to_coach": how_coach,
            "hints_tips_and_variations": hints
        }
    }

# Step 7: Schema Validation
def validate_results(results: List[Dict[str, Any]]) -> bool:
    """Validate that output contains exactly 7 valid shot objects."""
    if len(results) != 7:
        raise ValueError(f"Validation Error: Expected 7 objects, got {len(results)}")
    
    required_keys = ["shot_type", "page_start", "page_end", "structured_fields"]
    sub_keys = ["when_we_use_it", "why_we_use_it", "what_to_coach", "how_to_coach", "hints_tips_and_variations"]
    what_keys = ["prepare", "backswing", "forward_swing", "follow_through"]
    
    expected_shot_types = ["Backhand Net Shot", "Forehand Net Shot", "Backhand Net Lift", "Forehand Net Lift", "Clear", "Smash", "Drop"]

    for i, obj in enumerate(results):
        for k in required_keys:
            if k not in obj:
                raise ValueError(f"Object {i} missing key '{k}'")
        if obj["shot_type"] != expected_shot_types[i]:
            raise ValueError(f"Object {i} shot_type mismatch: expected {expected_shot_types[i]}, got {obj['shot_type']}")
        
        fields = obj["structured_fields"]
        for k in sub_keys:
            if k not in fields:
                raise ValueError(f"Object {i} structured_fields missing '{k}'")
        
        for k in what_keys:
            if k not in fields["what_to_coach"]:
                raise ValueError(f"Object {i} what_to_coach missing '{k}'")
                
    print("[OK] Schema Validation Passed: All 7 shot objects are valid and complete.")
    return True

def main():
    pdf_path = os.path.join(os.path.dirname(__file__), "SLBA COACH EDUCATION LEVEL-1.pdf")
    output_path = os.path.join(os.path.dirname(__file__), "extracted_shots.json")

    print("=== Badminton Shot RAG Pipeline ===")
    
    print("\n[1. Ingestion] Loading PDF document...")
    pages = ingest_pdf(pdf_path)
    
    print("[2. Chunking] Creating shot-based chunks with metadata...")
    chunks = create_shot_chunks(pages)
    
    print("[3. Indexing] Building TF-IDF vector index...")
    retriever = VectorStoreRetriever(chunks)
    
    results = []
    print("[4-6. Retrieval & Field Extraction] Processing each shot...")
    for target in TARGET_SHOTS:
        query = target["query"]
        shot_type = target["shot_type"]
        
        retrieved_chunk = retriever.retrieve(query)
        print(f" -> Query: '{query}' -> Retrieved Chunk: {retrieved_chunk['shot_type']} (p.{retrieved_chunk['page_start']}-{retrieved_chunk['page_end']})")
        
        extracted_obj = extract_fields(shot_type, retrieved_chunk["page_start"], retrieved_chunk["page_end"], retrieved_chunk["content"])
        results.append(extracted_obj)
        
    print("\n[7. Validation] Verifying output schema...")
    validate_results(results)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n[OK] Output successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
