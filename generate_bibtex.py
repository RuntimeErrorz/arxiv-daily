#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate BibTeX entries for approved papers from arxiv_papers.json
处理已发表和预印本论文，支持外部查询更新发表状态
"""

import json
import re
import requests
from datetime import datetime
from typing import Dict, List, Tuple, Optional


def clean_title(title: str) -> str:
    """清理标题用于BibTeX格式"""
    title = title.replace('\n', ' ').replace('\t', ' ')
    title = re.sub(r'\s+', ' ', title)
    return title.strip()


def format_authors(authors: List[str]) -> str:
    """格式化作者列表"""
    if not authors:
        return ""

    formatted_authors = []
    for author in authors:
        parts = author.strip().split()
        if len(parts) >= 2:
            last_name = parts[-1]
            first_names = " ".join(parts[:-1])
            formatted_authors.append(f"{last_name}, {first_names}")
        else:
            formatted_authors.append(author.strip())

    return " and ".join(formatted_authors)


def extract_arxiv_id(entry_id: str) -> Optional[str]:
    """从entry_id URL提取arXiv ID"""
    match = re.search(r'abs/(\d+\.\d+)', entry_id)
    if match:
        return match.group(1)
    return None


def generate_bibtex_key(arxiv_id: str, authors: List[str], year: str) -> str:
    """生成BibTeX键"""
    if authors and len(authors) > 0:
        first_author_last = authors[0].split()[-1].lower()
        first_author_last = re.sub(r'[^a-z0-9]', '', first_author_last)
        return f"{first_author_last}{year}_{arxiv_id.replace('.', '')}"
    else:
        return f"arxiv{year}_{arxiv_id.replace('.', '')}"


def parse_date(date_str: str) -> str:
    """解析日期字符串并返回年份"""
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return str(dt.year)
    except:
        return "2025"


def parse_journal_info(journal_ref: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """解析journal_ref信息，提取期刊名、卷号、页码等"""
    if not journal_ref:
        return None, None, None

    journal_ref = journal_ref.strip().replace('\n', ' ')
    journal_ref = re.sub(r'\s+', ' ', journal_ref)

    # 常见会议和期刊模式
    conference_patterns = [
        r'(CVPR|ICCV|ECCV|NeurIPS|ICML|ICLR|AAAI|IJCAI)',
        r'Conference on Computer Vision and Pattern Recognition',
        r'International Conference on Computer Vision',
        r'European Conference on Computer Vision',
        r'Conference on Neural Information Processing Systems',
        r'International Conference on Machine Learning',
        r'International Conference on Learning Representations',
    ]

    # 检查是否是会议
    for pattern in conference_patterns:
        if re.search(pattern, journal_ref, re.IGNORECASE):
            # 提取年份
            year_match = re.search(r'(\d{4})', journal_ref)
            year = year_match.group(1) if year_match else None
            return journal_ref, None, year

    # 尝试解析期刊格式
    # 格式如: "IEEE Transactions on..., vol. 61, pp. 1-16, 2023"
    ieee_match = re.search(
        r'(IEEE\s+[^,]+),?\s*vol\.\s*(\d+),?\s*pp\.\s*([^,]+),?\s*(\d{4})', journal_ref)
    if ieee_match:
        journal = ieee_match.group(1)
        volume = ieee_match.group(2)
        pages = ieee_match.group(3)
        year = ieee_match.group(4)
        return journal, volume, year

    # 其他期刊格式
    # 格式如: "Journal Name, Volume(Issue), Pages (Year)"
    general_match = re.search(
        r'([^,]+),?\s*(\d+)(?:\(\d+\))?,?\s*([^,\(]+)\s*\((\d{4})\)', journal_ref)
    if general_match:
        journal = general_match.group(1)
        volume = general_match.group(2)
        pages = general_match.group(3)
        year = general_match.group(4)
        return journal, volume, year

    return journal_ref, None, None


def query_semantic_scholar(arxiv_id: str, max_retries: int = 5) -> Optional[Dict]:
    """查询Semantic Scholar API获取发表信息，处理429错误重试"""
    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
    params = {
        'fields': 'title,authors,venue,year,journal,publicationDate,externalIds'
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            print(response.url)  # 打印请求的URL
            if response.status_code == 200:
                data = response.json()
                return data
            elif response.status_code == 429:
                # Rate limit exceeded, retry immediately
                print(
                    f"Rate limit hit for {arxiv_id}, retry {attempt + 1}/{max_retries}")
                continue
            elif response.status_code == 404:
                # Paper not found, no point in retrying
                print(f"Paper {arxiv_id} not found in Semantic Scholar")
                return None
            else:
                print(
                    f"Semantic Scholar API error for {arxiv_id}: {response.status_code}")
                if attempt < max_retries - 1:
                    continue
                return None

        except requests.exceptions.RequestException as e:
            print(
                f"Network error for {arxiv_id}, attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                continue
            return None
        except Exception as e:
            print(
                f"Unexpected error querying Semantic Scholar for {arxiv_id}: {e}")
            return None

    print(f"Max retries exceeded for {arxiv_id}")
    return None


def query_external_sources(arxiv_id: str) -> Optional[Dict]:
    """查询多个外部数据源获取发表信息"""
    # 首先尝试 Semantic Scholar
    result = query_semantic_scholar(arxiv_id, max_retries=1000000000000)
    if result:
        return result

    # 如果失败，可以尝试其他数据源
    # result = query_crossref(arxiv_id)
    # if result:
    #     return result

    return None


def generate_bibtex_entry(paper_id: str, paper_data: Dict) -> str:
    """生成单个BibTeX条目，完全基于外部查询结果"""
    title = clean_title(paper_data.get('title', ''))
    authors = paper_data.get('authors', [])
    published = paper_data.get('published', '')
    arxiv_id = extract_arxiv_id(paper_data.get('entry_id', '')) or paper_id

    year = parse_date(published)
    authors_formatted = format_authors(authors)
    bibtex_key = generate_bibtex_key(arxiv_id, authors, year)

    # 总是查询外部数据源获取最新发表信息
    print(f"Querying external sources for {arxiv_id}...")
    external_info = query_external_sources(arxiv_id)

    venue = external_info.get('venue') if external_info else None
    ext_year = external_info.get('year') if external_info else None
    ext_title = external_info.get('title') if external_info else None
    ext_authors = external_info.get('authors') if external_info else None
    
    # 使用外部查询的信息更新数据
    if ext_title and ext_title.strip():
        title = clean_title(ext_title)
    
    if ext_authors and isinstance(ext_authors, list):
        # Semantic Scholar返回的作者格式可能不同，需要处理
        author_names = []
        for author in ext_authors:
            if isinstance(author, dict) and 'name' in author:
                author_names.append(author['name'])
            elif isinstance(author, str):
                author_names.append(author)
        if author_names:
            authors_formatted = format_authors(author_names)
            bibtex_key = generate_bibtex_key(arxiv_id, author_names, str(ext_year) if ext_year else year)
    
    if ext_year:
        year = str(ext_year)

    has_publication_info = (venue 
                           and venue.strip() 
                           and venue != "arXiv.org")
    
    if has_publication_info:
        # 已发表论文 - 使用外部查询的venue信息
        journal_ref = venue
        journal, volume, pub_year = parse_journal_info(journal_ref)
        if pub_year:
            year = pub_year

        # 判断是会议还是期刊 - 优先使用journal.name，然后是venue
        venue_info = journal_ref or ''
        journal_name = ''
        if external_info and 'journal' in external_info and external_info['journal']:
            journal_name = external_info['journal'].get('name', '')
        
        # 检查的文本包括journal.name和venue
        text_to_check = f"{venue_info} {journal_name}".lower()
        
        # 更全面的会议关键词检查
        conference_keywords = [
            'conference', 'proceedings', 'workshop', 'symposium',
            'cvpr', 'iccv', 'eccv', 'neurips', 'icml', 'iclr', 'aaai', 'ijcai',
            'acm', 'ieee conference', 'international conference'
        ]
        
        # 期刊关键词
        journal_keywords = [
            'journal', 'transactions', 'ieee trans', 'acm trans', 'nature', 'science'
        ]
        
        # 首先检查是否包含明确的会议关键词
        is_conference = any(keyword in text_to_check for keyword in conference_keywords)
        
        # 如果没有明确的会议关键词，检查是否有期刊关键词
        if not is_conference:
            has_journal_keywords = any(keyword in text_to_check for keyword in journal_keywords)
            if not has_journal_keywords:
                # 如果都没有明确关键词，通过DBLP的信息判断
                if external_info and 'externalIds' in external_info:
                    dblp_id = external_info['externalIds'].get('DBLP', '')
                    if 'conf/' in dblp_id:
                        is_conference = True
                    elif 'journals/' in dblp_id:
                        is_conference = False

        if is_conference:
            entry_type = "inproceedings"
            bibtex_entry = f"@{entry_type}{{{bibtex_key},\n"
            bibtex_entry += f"  title = {{{title}}},\n"
            if authors_formatted:
                bibtex_entry += f"  author = {{{authors_formatted}}},\n"
            bibtex_entry += f"  booktitle = {{{journal_ref}}},\n"
            bibtex_entry += f"  year = {{{year}}},\n"
            # 从外部信息获取DOI
            ext_doi = None
            if external_info and 'externalIds' in external_info:
                ext_doi = external_info['externalIds'].get('DOI')
            if ext_doi:
                bibtex_entry += f"  doi = {{{ext_doi}}},\n"
            bibtex_entry += f"  eprint = {{{arxiv_id}}},\n"
            bibtex_entry += f"  archivePrefix = {{arXiv}},\n"
            bibtex_entry += f"  primaryClass = {{{paper_data.get('primary_category', 'cs.CV')}}}\n"
        else:
            entry_type = "article"
            bibtex_entry = f"@{entry_type}{{{bibtex_key},\n"
            bibtex_entry += f"  title = {{{title}}},\n"
            if authors_formatted:
                bibtex_entry += f"  author = {{{authors_formatted}}},\n"
            bibtex_entry += f"  journal = {{{journal or journal_ref}}},\n"
            if volume:
                bibtex_entry += f"  volume = {{{volume}}},\n"
            bibtex_entry += f"  year = {{{year}}},\n"
            # 从外部信息获取DOI
            ext_doi = None
            if external_info and 'externalIds' in external_info:
                ext_doi = external_info['externalIds'].get('DOI')
            if ext_doi:
                bibtex_entry += f"  doi = {{{ext_doi}}},\n"
            bibtex_entry += f"  eprint = {{{arxiv_id}}},\n"
            bibtex_entry += f"  archivePrefix = {{arXiv}},\n"
            bibtex_entry += f"  primaryClass = {{{paper_data.get('primary_category', 'cs.CV')}}}\n"
    else:
        # 预印本
        entry_type = "misc"
        bibtex_entry = f"@{entry_type}{{{bibtex_key},\n"
        bibtex_entry += f"  title = {{{title}}},\n"
        if authors_formatted:
            bibtex_entry += f"  author = {{{authors_formatted}}},\n"
        bibtex_entry += f"  year = {{{year}}},\n"
        bibtex_entry += f"  eprint = {{{arxiv_id}}},\n"
        bibtex_entry += f"  archivePrefix = {{arXiv}},\n"
        bibtex_entry += f"  primaryClass = {{{paper_data.get('primary_category', 'cs.CV')}}},\n"
        bibtex_entry += f"  note = {{arXiv preprint}}\n"

    bibtex_entry += "}"
    return bibtex_entry


def main():
    """主函数"""
    print("加载 arxiv_papers.json...")

    try:
        with open('arxiv_papers.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("错误：找不到 arxiv_papers.json 文件！")
        return
    except json.JSONDecodeError as e:
        print(f"错误：JSON格式无效 - {e}")
        return

    approved_papers = []
    total_papers = 0

    print("扫描已批准的论文...")

    # 遍历所有类别和论文
    for category, papers in data.items():
        if isinstance(papers, dict):
            for paper_id, paper_data in papers.items():
                total_papers += 1
                if isinstance(paper_data, dict) and paper_data.get('llm_approved') is True:
                    approved_papers.append((paper_id, paper_data, category))

    print(f"在 {total_papers} 篇论文中找到 {len(approved_papers)} 篇已批准的论文")

    if not approved_papers:
        print("没有找到已批准的论文！")
        return

    print("将为所有论文查询外部数据源获取最新发表状态...")

    # 生成BibTeX条目
    bibtex_entries = []
    published_count = 0
    preprint_count = 0
    updated_count = 0

    print("生成BibTeX条目...")

    for i, (paper_id, paper_data, category) in enumerate(approved_papers):
        print(f"处理论文 {i+1}/{len(approved_papers)}: {paper_id}")

        try:
            bibtex_entry = generate_bibtex_entry(paper_id, paper_data)
            bibtex_entries.append(bibtex_entry)

            # 统计发表状态（基于外部查询结果）
            # 这里我们需要重新查询来统计，或者从bibtex_entry的类型判断
            if "@inproceedings" in bibtex_entry or "@article" in bibtex_entry:
                published_count += 1
            else:
                preprint_count += 1

        except Exception as e:
            print(f"处理论文 {paper_id} 时出错: {e}")
            continue

    # 写入文件
    output_file = 'approved_papers_bibtex.bib'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("% BibTeX entries for approved arXiv papers\n")
        f.write(
            f"% Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"% Total approved papers: {len(approved_papers)}\n")
        f.write(f"% Published papers: {published_count}\n")
        f.write(f"% Preprints: {preprint_count}\n")
        f.write(f"% All data based on external API queries\n")
        f.write("\n")

        for entry in bibtex_entries:
            f.write(entry + "\n\n")

    print(f"\n成功生成 {len(bibtex_entries)} 个BibTeX条目")
    print(f"已发表论文: {published_count}")
    print(f"预印本: {preprint_count}")
    print(f"所有数据基于外部API查询结果")
    print(f"输出保存到: {output_file}")


if __name__ == "__main__":
    main()
