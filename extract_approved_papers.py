#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to extract title and abstract from arxiv_papers.json for papers with llm_approved=True
"""

import json
import os
from datetime import datetime

def extract_approved_papers(input_file='arxiv_papers.json', output_file='approved_papers.json'):
    """
    Extract papers with llm_approved=True and save their title and abstract
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to the output JSON file
    """
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return
    
    print(f"Loading data from {input_file}...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file: {e}")
        return
    except Exception as e:
        print(f"Error: Failed to read file: {e}")
        return
    
    approved_papers = []
    total_papers = 0
    approved_count = 0
    
    # Iterate through all categories and papers
    for category, papers in data.items():
        if isinstance(papers, dict):
            for paper_id, paper_info in papers.items():
                total_papers += 1
                
                # Check if the paper is approved by LLM
                if paper_info.get('llm_approved', False) == True:
                    approved_count += 1
                    
                    # Extract title and abstract (summary)
                    paper_data = {
                        'paper_id': paper_id,
                        'category': category,
                        'title': paper_info.get('title', ''),
                        'abstract': paper_info.get('summary', ''),
                        'venue': paper_info.get('venue_info', {}).get('venue', ''),
                        'authors': paper_info.get('authors', []),
                        'published': paper_info.get('published', ''),
                        'pdf_url': paper_info.get('pdf_url', '')
                    }
                    
                    approved_papers.append(paper_data)
    
    print(f"Found {approved_count} approved papers out of {total_papers} total papers")
    
    # Prepare output data
    output_data = {
        'extraction_date': datetime.now().isoformat(),
        'total_papers_count': total_papers,
        'approved_papers_count': approved_count,
        'approved_papers': approved_papers
    }
    
    # Save to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved {approved_count} approved papers to {output_file}")
    except Exception as e:
        print(f"Error: Failed to save output file: {e}")
        return
    
    # Also create a markdown file for better readability
    md_output_file = output_file.replace('.json', '.md')
    try:
        with open(md_output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Approved Papers from arXiv\n\n")
            f.write(f"**Extraction Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Papers:** {total_papers}\n\n")
            f.write(f"**Approved Papers:** {approved_count}\n\n")
            f.write("---\n\n")
            
            for i, paper in enumerate(approved_papers, 1):
                f.write(f"## {i}. {paper['title']}\n\n")
                f.write(f"**Paper ID:** {paper['paper_id']}\n\n")
                f.write(f"**Category:** {paper['category']}\n\n")
                f.write(f"**Venue:** {paper['venue']}\n\n")
                f.write(f"**Authors:** {', '.join(paper['authors'])}\n\n")
                f.write(f"**Published:** {paper['published']}\n\n")
                f.write(f"**PDF URL:** {paper['pdf_url']}\n\n")
                f.write(f"**Abstract:**\n{paper['abstract']}\n\n")
                f.write("---\n\n")
        
        print(f"Also created markdown version: {md_output_file}")
    except Exception as e:
        print(f"Warning: Failed to create markdown file: {e}")

def main():
    """Main function"""
    print("=" * 60)
    print("Extracting LLM-approved papers from arxiv_papers.json")
    print("=" * 60)
    
    # Run the extraction
    extract_approved_papers()
    
    print("\nExtraction completed!")

if __name__ == "__main__":
    main()
