#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate BibTeX entries for approved papers from arxiv_papers.json
使用 CrossRef DOI 内容协商接口获取标准 BibTeX 格式
对于没有 DOI 的论文，使用 Semantic Scholar API 获取信息
"""

import json
import re
import requests
from datetime import datetime
from typing import Dict, List, Optional
import time

# 添加重试配置
MAX_RETRIES = 100000000000
RETRY_DELAY = 0  # 秒


def extract_arxiv_id(entry_id: str) -> Optional[str]:
    """从 entry_id URL 提取 arXiv ID"""
    match = re.search(r'abs/(\d+\.\d+)', entry_id)
    return match.group(1) if match else None


def clean_title(title: str) -> str:
    """清理标题格式"""
    return re.sub(r'\s+', ' ', title.replace('\n', ' ').replace('\t', ' ')).strip()


def format_authors(authors: List[str]) -> str:
    """格式化作者列表为 BibTeX 格式"""
    if not authors:
        return ""

    formatted = []
    for author in authors:
        parts = author.strip().split()
        if len(parts) >= 2:
            formatted.append(f"{parts[-1]}, {' '.join(parts[:-1])}")
        else:
            formatted.append(author.strip())

    return " and ".join(formatted)


def generate_bibtex_key(arxiv_id: str, authors: List[str], year: str, title: str = "", used_keys: set = None) -> str:
    """生成 BibTeX 引用键 - 格式：作者姓氏+年份+关键词，确保唯一性"""
    if used_keys is None:
        used_keys = set()
    
    if authors:
        first_author = re.sub(r'[^a-z0-9]', '', authors[0].split()[-1].lower())

        # 从标题中提取关键词
        if title:
            # 移除常见停用词和符号
            title_clean = re.sub(r'[^\w\s]', ' ', title.lower())
            words = title_clean.split()

            # 停用词列表
            stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                          'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                          'to', 'was', 'will', 'with', 'via', 'using', 'based', 'through'}

            # 提取关键词（长度>=3且不是停用词）
            keywords = [word for word in words if len(
                word) >= 3 and word not in stop_words]

            if keywords:
                # 尝试不同数量的关键词组合
                for num_keywords in range(1, min(6, len(keywords) + 1)):
                    selected_keywords = keywords[:num_keywords]
                    keyword_part = ''.join(selected_keywords)
                    candidate_key = f"{first_author}{year}{keyword_part}"
                    
                    if candidate_key not in used_keys:
                        used_keys.add(candidate_key)
                        return candidate_key
                    
                    if num_keywords > 1:
                        print(f"  Key '{candidate_key}' already used, trying {num_keywords + 1} keywords...")

        # 如果没有合适的关键词或都重复，回退到 arxiv_id
        base_key = f"{first_author}{year}_{arxiv_id.replace('.', '')}"
    else:
        base_key = f"arxiv{year}_{arxiv_id.replace('.', '')}"
    
    # 检查基础键是否重复
    if base_key not in used_keys:
        used_keys.add(base_key)
        return base_key
    
    # 如果基础键也重复，添加数字后缀
    counter = 1
    while f"{base_key}_{counter}" in used_keys:
        counter += 1
    
    final_key = f"{base_key}_{counter}"
    used_keys.add(final_key)
    print(f"  Using numbered key: {final_key}")
    return final_key


def parse_year(date_str: str) -> str:
    """从日期字符串提取年份"""
    try:
        return str(datetime.fromisoformat(date_str.replace('Z', '+00:00')).year)
    except:
        return "2025"


def query_doi_bibtex(doi: str, max_retries: int = MAX_RETRIES) -> Optional[str]:
    """使用 CrossRef 内容协商获取 BibTeX"""
    headers = {
        'Accept': 'application/x-bibtex',
        'User-Agent': 'arxiv-daily-bibtex-generator (mailto:example@email.com)'
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(
                f"https://doi.org/{doi}", headers=headers, timeout=10)
            print(f"DOI query: {response.url} -> {response.status_code}")

            if response.status_code == 200:
                content = response.text.strip()
                if content and content.startswith('@'):
                    return content
            elif response.status_code == 404:
                print(f"DOI {doi} not found")
                return None  # 404不需要重试
            elif response.status_code == 429:
                print(f"Rate limited, waiting {RETRY_DELAY * (attempt + 1)} seconds...")
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            else:
                print(f"DOI error: {response.status_code}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {RETRY_DELAY} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(RETRY_DELAY)
                    continue

        except requests.exceptions.ConnectionError as e:
            print(f"Connection error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {RETRY_DELAY * (attempt + 1)} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
        except requests.exceptions.Timeout as e:
            print(f"Timeout error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {RETRY_DELAY} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(RETRY_DELAY)
                continue
        except Exception as e:
            print(f"DOI query failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {RETRY_DELAY} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(RETRY_DELAY)
                continue

    print(f"Failed to query DOI {doi} after {max_retries} attempts")
    return None


def calculate_title_similarity(title1: str, title2: str) -> float:
    """计算两个标题的相似度"""
    # 简单的相似度算法：基于共同词汇的比例
    def normalize_title(title):
        # 转小写，移除标点，分词
        title = re.sub(r'[^\w\s]', ' ', title.lower())
        return set(word for word in title.split() if len(word) > 2)

    words1 = normalize_title(title1)
    words2 = normalize_title(title2)

    if not words1 or not words2:
        return 0.0

    # 计算 Jaccard 相似度
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))

    return intersection / union if union > 0 else 0.0


def query_crossref_search(title: str, authors: List[str], max_retries: int = MAX_RETRIES) -> Optional[str]:
    """使用 CrossRef API 模糊检索获取 DOI"""
    url = "https://api.crossref.org/works"

    # 构建查询参数
    params = {}
    params['query.bibliographic'] = title
    params['rows'] = '10'
    params['select'] = 'DOI,title,author,published-print,published-online'

    # 处理作者信息
    if authors:
        first_author = authors[0] if authors[0] else ""
        if first_author:
            author_parts = first_author.split()
            if author_parts:
                last_name = author_parts[-1]
                clean_author = last_name
                if clean_author and len(clean_author) >= 2:
                    params['query.author'] = clean_author

    headers = {
        'User-Agent': 'arxiv-daily-bibtex-generator (mailto:example@email.com)',
        'Accept': 'application/json',
        'Accept-Charset': 'utf-8'
    }

    print(f"  Query params: title='{params.get('query.bibliographic', '')[:50]}...', author='{params.get('query.author', 'N/A')}'")

    for attempt in range(max_retries):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=15,
                allow_redirects=True
            )

            print(f"CrossRef search: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                items = data.get('message', {}).get('items', [])

                if items:
                    best_match = None
                    best_similarity = 0.0

                    # 遍历所有结果，找到标题最相似的
                    for item in items:
                        item_title = item.get('title', [''])[0] if item.get('title') else ''
                        if item_title:
                            similarity = calculate_title_similarity(title, item_title)
                            print(f"  Candidate: {item_title[:60]}... (similarity: {similarity:.3f})")

                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match = item

                    # 只有相似度超过阈值才认为匹配
                    if best_match and best_similarity > 0.8:
                        doi = best_match.get('DOI')
                        if doi:
                            found_title = best_match.get('title', [''])[0] if best_match.get('title') else ''
                            print(f"  [MATCH] Best match: {found_title[:60]}... (similarity: {best_similarity:.3f})")
                            print(f"  DOI: {doi}")
                            return doi

                    print(f"  [NO MATCH] No good match found (best similarity: {best_similarity:.3f})")
                    return None
                else:
                    print("  No matching papers found")
                    return None
            elif response.status_code == 429:
                print(f"Rate limited, waiting {RETRY_DELAY * (attempt + 1)} seconds...")
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            else:
                print(f"  CrossRef search error: {response.status_code}")
                if attempt < max_retries - 1:
                    print(f"  Retrying in {RETRY_DELAY} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(RETRY_DELAY)
                    continue

        except requests.exceptions.ConnectionError as e:
            print(f"  CrossRef connection error: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {RETRY_DELAY * (attempt + 1)} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
        except requests.exceptions.Timeout as e:
            print(f"  CrossRef timeout error: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {RETRY_DELAY} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(RETRY_DELAY)
                continue
        except Exception as e:
            print(f"  CrossRef search failed: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {RETRY_DELAY} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(RETRY_DELAY)
                continue

    print(f"  Failed to search CrossRef after {max_retries} attempts")
    return None


def query_semantic_scholar(arxiv_id: str = None, max_retries: int = MAX_RETRIES) -> Optional[Dict]:
    """使用 Semantic Scholar API 搜索论文信息"""

    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
    headers = {
        'User-Agent': 'arxiv-daily-bibtex-generator (mailto:example@email.com)'
    }
    params = {
        'fields': 'title,authors,year,venue,publicationVenue,url,abstract,citationCount,externalIds,publicationDate,journal,citationStyles'
    }

    print(f"  Semantic Scholar query by arXiv ID: arXiv:{arxiv_id}")

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)
            print(f"Semantic Scholar (arXiv ID): {response.status_code}")

            if response.status_code == 200:
                paper = response.json()
                if paper and paper.get('title'):
                    found_title = paper.get('title', '')
                    print(f"  [MATCH] Found by arXiv ID: {found_title[:60]}...")
                    return paper
                else:
                    print(f"  [NO DATA] arXiv ID found but no complete data")
                    return None
            elif response.status_code == 404:
                print(f"  [NOT FOUND] arXiv:{arxiv_id} not found in Semantic Scholar")
                return None  # 404不需要重试
            elif response.status_code == 429:
                print(f"  Rate limited, waiting {RETRY_DELAY * (attempt + 1)} seconds...")
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            else:
                print(f"  Semantic Scholar (arXiv ID) error: {response.status_code}")
                if attempt < max_retries - 1:
                    print(f"  Retrying in {RETRY_DELAY} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(RETRY_DELAY)
                    continue

        except requests.exceptions.ConnectionError as e:
            print(f"  Semantic Scholar connection error: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {RETRY_DELAY * (attempt + 1)} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
        except requests.exceptions.Timeout as e:
            print(f"  Semantic Scholar timeout error: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {RETRY_DELAY} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(RETRY_DELAY)
                continue
        except Exception as e:
            print(f"  Semantic Scholar (arXiv ID) search failed: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {RETRY_DELAY} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(RETRY_DELAY)
                continue

    print(f"  Failed to query Semantic Scholar after {max_retries} attempts")
    return None


def extract_venue_from_semantic_scholar(paper_info: Dict) -> str:
    """从Semantic Scholar数据中提取venue信息"""
    # 优先使用publicationVenue
    publication_venue = paper_info.get('publicationVenue')
    if publication_venue and isinstance(publication_venue, dict):
        venue_name = publication_venue.get('name', '')
        if venue_name:
            return venue_name
    
    # 回退到venue字段
    venue = paper_info.get('venue', '')
    if venue:
        return venue
    
    # 如果是期刊文章，尝试从journal字段获取
    journal = paper_info.get('journal')
    if journal and isinstance(journal, dict):
        journal_name = journal.get('name', '')
        if journal_name:
            return journal_name
    
    return ''


def extract_doi_from_semantic_scholar(paper_info: Dict) -> str:
    """从Semantic Scholar数据中提取DOI"""
    external_ids = paper_info.get('externalIds', {})
    if external_ids and isinstance(external_ids, dict):
        doi = external_ids.get('DOI', '')
        if doi:
            return doi
    
    return ''

def generate_bibtex_from_semantic_scholar(paper_info: Dict, bibtex_key: str, arxiv_id: str) -> str:
    """从Semantic Scholar信息生成BibTeX条目"""
    title = paper_info.get('title', '')
    authors = paper_info.get('authors', [])
    year = paper_info.get('year')
    
    # 使用新的venue提取函数
    venue = extract_venue_from_semantic_scholar(paper_info)
    
    # 使用新的DOI提取函数
    doi = extract_doi_from_semantic_scholar(paper_info)
    
    url = paper_info.get('url', '')

    # 格式化作者
    author_names = []
    for author in authors:
        if isinstance(author, dict):
            name = author.get('name', '')
        else:
            name = str(author)
        if name:
            author_names.append(name)

    authors_formatted = format_authors(author_names)
    
    # 直接从API获取venue类型，不需要复杂的字符串匹配
    publication_venue = paper_info.get('publicationVenue', {})
    if publication_venue and isinstance(publication_venue, dict):
        venue_type = publication_venue.get('type', '').lower()
    else:
        # 如果没有publicationVenue信息，简单判断是否为期刊
        venue_type = "conference" if "workshop" in paper_info.get('venue', '').lower() else "journal"

    # 根据类型生成BibTeX条目
    if venue_type == "journal":
        entry_type = "article"
        entry = f"@{entry_type}{{{bibtex_key},\n"
        entry += f"  title = {{{title}}},\n"
        if authors_formatted:
            entry += f"  author = {{{authors_formatted}}},\n"
        entry += f"  journal = {{{venue}}},\n"
        if year:
            entry += f"  year = {{{year}}},\n"
        
        # 添加期刊特有信息
        journal_info = paper_info.get('journal', {})
        if journal_info and isinstance(journal_info, dict):
            volume = journal_info.get('volume', '')
            pages = journal_info.get('pages', '')
            if volume:
                entry += f"  volume = {{{volume}}},\n"
            if pages:
                entry += f"  pages = {{{pages}}},\n"
        
        if doi:
            entry += f"  doi = {{{doi}}},\n"
        if url:
            entry += f"  url = {{{url}}},\n"
        entry += f"  eprint = {{{arxiv_id}}},\n"
        entry += f"  archivePrefix = {{arXiv}}\n"
        entry += "}"
        
    elif venue_type == "conference":
        entry_type = "inproceedings"
        entry = f"@{entry_type}{{{bibtex_key},\n"
        entry += f"  title = {{{title}}},\n"
        if authors_formatted:
            entry += f"  author = {{{authors_formatted}}},\n"
        entry += f"  booktitle = {{{venue}}},\n"
        if year:
            entry += f"  year = {{{year}}},\n"
        if doi:
            entry += f"  doi = {{{doi}}},\n"
        if url:
            entry += f"  url = {{{url}}},\n"
        entry += f"  eprint = {{{arxiv_id}}},\n"
        entry += f"  archivePrefix = {{arXiv}}\n"
        entry += "}"
        
    else:
        # 默认为misc类型
        entry_type = "misc"
        entry = f"@{entry_type}{{{bibtex_key},\n"
        entry += f"  title = {{{title}}},\n"
        if authors_formatted:
            entry += f"  author = {{{authors_formatted}}},\n"
        if venue:
            entry += f"  howpublished = {{{venue}}},\n"
        if year:
            entry += f"  year = {{{year}}},\n"
        if doi:
            entry += f"  doi = {{{doi}}},\n"
        if url:
            entry += f"  url = {{{url}}},\n"
        entry += f"  eprint = {{{arxiv_id}}},\n"
        entry += f"  archivePrefix = {{arXiv}}\n"
        entry += "}"

    return entry


def post_process_bibtex(bibtex_entry: str) -> str:
    """后处理BibTeX条目，规范化格式"""
    # 确保输入不为None
    if not bibtex_entry:
        return bibtex_entry
    
    # 1. 将所有 @inbook 转换为 @inproceedings
    bibtex_entry = re.sub(r'@inbook\{', '@inproceedings{', bibtex_entry)
    
    # 2. 检查特定的期刊/会议，转换为相应类型
    if 'Proceedings of the AAAI Conference on Artificial Intelligence' in bibtex_entry:
        # 确保是 @inproceedings 类型
        bibtex_entry = re.sub(r'@article\{', '@inproceedings{', bibtex_entry)        
        # 将 journal 字段转换为 booktitle
        bibtex_entry = re.sub(r'journal\s*=\s*\{Proceedings of the AAAI Conference on Artificial Intelligence\}', 
                             'booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence}', 
                             bibtex_entry)
    
    return bibtex_entry


def generate_bibtex_entry(paper_id: str, paper_data: Dict, used_keys: set) -> str:
    """生成 BibTeX 条目"""
    arxiv_id = extract_arxiv_id(paper_data.get('entry_id', '')) or paper_id
    authors = paper_data.get('authors', [])
    year = parse_year(paper_data.get('published', ''))
    title = clean_title(paper_data.get('title', ''))
    
    # 生成唯一的 BibTeX 引用键
    bibtex_key = generate_bibtex_key(arxiv_id, authors, year, title, used_keys)

    print(f"Processing {arxiv_id}...")
    print(f"  Title: {title[:60]}...")
    print(f"  BibTeX key: {bibtex_key}")

    # 步骤1: 尝试 CrossRef 模糊检索
    found_doi = query_crossref_search(title, authors)

    if found_doi:
        # 步骤2: 使用找到的 DOI 进行内容协商
        print(f"  Trying DOI content negotiation for: {found_doi}")
        bibtex_content = query_doi_bibtex(found_doi)

        if bibtex_content:
            # 替换引用键
            key_match = re.search(r'@\w+\{([^,]+),', bibtex_content)
            if key_match:
                original_key = key_match.group(1)
                bibtex_content = bibtex_content.replace(
                    f'{{{original_key},', f'{{{bibtex_key},', 1)

            # 添加 arXiv 信息（如果没有）
            # if 'eprint' not in bibtex_content.lower():
            #     insert_pos = bibtex_content.rfind('}')
            #     if insert_pos != -1:
            #         arxiv_fields = f",\n  eprint = {{{arxiv_id}}},\n  archivePrefix = {{arXiv}},\n  primaryClass = {{{paper_data.get('primary_category', 'cs.CV')}}}\n"
            #         bibtex_content = bibtex_content[:insert_pos] + \
            #             arxiv_fields + bibtex_content[insert_pos:]

            print(
                f"[SUCCESS] Found publication via CrossRef search for {arxiv_id}")
            return bibtex_content

    # 步骤3: 尝试 Semantic Scholar 搜索（优先使用 arXiv ID）
    print(f"  Trying Semantic Scholar search...")
    semantic_paper = query_semantic_scholar(arxiv_id)

    if semantic_paper:
        # 使用新的venue提取函数
        venue = extract_venue_from_semantic_scholar(semantic_paper)
        
        # 检查是否有有效的venue信息
        if venue and venue.strip():
            print(f"  Generating BibTeX from Semantic Scholar data...")
            bibtex_content = generate_bibtex_from_semantic_scholar(
                semantic_paper, bibtex_key, arxiv_id)

            if bibtex_content:
                print(
                    f"[SUCCESS] Found publication via Semantic Scholar for {arxiv_id} (venue: {venue}")
                return bibtex_content
        else:
            print(f"  [NO VENUE] Semantic Scholar found paper but no publication venue, skipping to arXiv DOI...")

    # 步骤4: 尝试直接使用 arXiv DOI（备选方案）
    print(f"  Trying arXiv DOI content negotiation...")
    arxiv_doi = f"10.48550/arXiv.{arxiv_id}"
    bibtex_content = query_doi_bibtex(arxiv_doi)

    if bibtex_content:
        # 替换引用键
        key_match = re.search(r'@\w+\{([^,]+),', bibtex_content)
        if key_match:
            original_key = key_match.group(1)
            bibtex_content = bibtex_content.replace(
                f'{{{original_key},', f'{{{bibtex_key},', 1)

        # 添加 arXiv 信息（如果没有）
        # if 'eprint' not in bibtex_content.lower():
        #     insert_pos = bibtex_content.rfind('}')
        #     if insert_pos != -1:
        #         arxiv_fields = f",\n  eprint = {{{arxiv_id}}},\n  archivePrefix = {{arXiv}},\n  primaryClass = {{{paper_data.get('primary_category', 'cs.CV')}}}\n"
        #         bibtex_content = bibtex_content[:insert_pos] + \
        #             arxiv_fields + bibtex_content[insert_pos:]

        print(f"[SUCCESS] Found arXiv publication for {arxiv_id}")
        return bibtex_content

    # 步骤5: 回退到手动生成预印本条目
    print(f"[PREPRINT] No publication found for {arxiv_id}, creating preprint entry")
    authors_formatted = format_authors(authors)
    
    entry = f"@misc{{{bibtex_key},\n"
    entry += f"  title = {{{title}}},\n"
    if authors_formatted:
        entry += f"  author = {{{authors_formatted}}},\n"
    entry += f"  year = {{{year}}},\n"
    entry += f"  eprint = {{{arxiv_id}}},\n"
    entry += f"  archivePrefix = {{arXiv}},\n"
    entry += f"  primaryClass = {{{paper_data.get('primary_category', 'cs.CV')}}},\n"
    entry += f"  note = {{arXiv preprint}}\n"
    entry += "}"
    
    return entry


def main():
    """主函数"""
    print("Loading arxiv_papers.json...")

    try:
        with open('arxiv_papers.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: arxiv_papers.json not found!")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}")
        return

    # 收集已批准论文
    approved_papers = []
    total_papers = 0

    for category, papers in data.items():
        if isinstance(papers, dict):
            for paper_id, paper_data in papers.items():
                total_papers += 1
                if isinstance(paper_data, dict) and paper_data.get('llm_approved') is True:
                    approved_papers.append((paper_id, paper_data, category))

    print(
        f"Found {len(approved_papers)} approved papers out of {total_papers} total papers")

    if not approved_papers:
        print("No approved papers found!")
        return

    # 生成 BibTeX 条目
    bibtex_entries = []
    published_count = 0
    preprint_count = 0
    used_keys = set()  # 跟踪已使用的引用键

    print("\nGenerating BibTeX entries...")

    for i, (paper_id, paper_data, category) in enumerate(approved_papers, 1):
        print(f"[{i}/{len(approved_papers)}] {paper_id}")

        try:
            entry = generate_bibtex_entry(paper_id, paper_data, used_keys)
            
            # 确保entry不为None才进行后处理
            entry = post_process_bibtex(entry)
            bibtex_entries.append(entry)

            # 统计
            if "@inproceedings" in entry or "@article" in entry:
                published_count += 1
            else:
                preprint_count += 1

        except Exception as e:
            print(f"Error processing {paper_id}: {e}")
            continue

    # 写入文件 - 明确指定 UTF-8 编码
    output_file = 'approved_papers_bibtex.bib'
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        f.write("% BibTeX entries for approved arXiv papers\n")
        f.write(
            f"% Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"% Total: {len(approved_papers)} papers\n")
        f.write(
            f"% Published: {published_count}, Preprints: {preprint_count}\n")
        f.write(f"% Data source: CrossRef DOI content negotiation\n")
        f.write("\n")

        for entry in bibtex_entries:
            f.write(entry + "\n\n")

    print(f"\n[COMPLETE] Generated {len(bibtex_entries)} BibTeX entries")
    print(f"  Published papers: {published_count}")
    print(f"  Preprints: {preprint_count}")
    print(f"  Output saved to: {output_file}")


if __name__ == "__main__":
    main()