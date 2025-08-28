import time
import arxiv
import os
import json
import google.generativeai as genai

from datetime import datetime, timezone, timedelta
from tqdm import tqdm

ARXIV_PAPERS_JSON = "arxiv_papers.json"
UPDATE_LOG = "update_log.json"
ARXIV_QUERY_CONFIG = "arxiv_query_config.json"
MODEL = "gemini-2.5-flash-lite"

def setup_gemini():
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("警告: 未找到GEMINI_API_KEY环境变量，Gemini筛选功能将被禁用")
        return None
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel(MODEL)
        print("Gemini模型初始化成功")
        return model
    except Exception as e:
        print(f"Gemini初始化失败: {e}")
        return None

def evaluate_paper_with_gemini(model, title, abstract, topic_prompt):
    if not model or not topic_prompt:
        return True
    prompt = f"""
            请根据以下标准判断这篇论文是否符合要求：

            评估标准：
            {topic_prompt}

            论文信息：
            标题：{title}
            摘要：{abstract}

            请使用简体中文只回答"是"或"否"，不需要解释。如果论文明确符合标准，回答"是"；如果不符合或不确定，回答"否"。
            """
    try:
        response = model.generate_content(prompt)
        result = response.text.strip().lower()
        return "是" in result or "yes" in result
    except Exception as e:
        print(f"Gemini评估失败: {e}")
        return True  # 出错时默认保留

def get_daily_papers(topic, query, max_results, config, json_file_name):
    try:
        with open(json_file_name, "r", encoding='utf-8') as f:
            content = f.read()
            if content.strip():
                old_data = json.loads(content)
            else:
                old_data = {}
    except (FileNotFoundError, json.JSONDecodeError):
        old_data = {}
        
    content = dict()
    topic_prompt = config.get(topic, {}).get('prompt', '')
    gemini_model = setup_gemini() if topic_prompt else None
    
    if topic_prompt and gemini_model:
        print(f"AI筛选已启用，标准: {topic_prompt[:50]}...")
    
    res = arxiv.Client(delay_seconds=3, num_retries=10, page_size=min(max_results, 9999)).results(
        arxiv.Search(
            query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate
        )
    )
    
    for result in tqdm(res, desc=f"Processing {topic}"):
        paper_id = result.entry_id.split('/')[-1].split('v')[0]
        
        paper_info = {
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'published': result.published.isoformat(),
            'updated': result.updated.isoformat(),
            'summary': result.summary,
            'categories': result.categories,
            'primary_category': result.primary_category,
            'pdf_url': result.pdf_url,
            'journal_ref': result.journal_ref,
            'doi': result.doi,
            'comment': result.comment,
            'entry_id': result.entry_id,
            'links': [
                {
                    'href': link.href,
                    'title': link.title,
                    'rel': link.rel,
                    'content_type': link.content_type
                } for link in result.links
            ]
        }
        
        if topic_prompt and gemini_model and paper_id not in old_data.get(topic, {}):
            is_relevant = evaluate_paper_with_gemini(
                gemini_model,
                paper_info['title'],
                paper_info['summary'],
                topic_prompt
            )
            paper_info['llm_approved'] = is_relevant
            print(f"Paper {paper_id}: {'✓' if is_relevant else '✗'} - {paper_info['title'][:60]}...")
            time.sleep(5)
    
            
        content[paper_id] = paper_info

    data = {topic: content}
    return data

def update_json_file(filename, data_all, cnt):
    with open(filename, "r", encoding='utf-8') as f:
        content = f.read()
        if not content:
            m = {}
        else:
            m = json.loads(content)
    json_data = m.copy()
    for data in data_all:
        for topic in data.keys():
            papers = data[topic]
            
            if topic not in json_data:
                json_data[topic] = {}
            
            new_papers_count = 0
            for paper_id, paper_info in papers.items():
                if paper_id not in json_data[topic]:
                    new_papers_count += 1
                    json_data[topic][paper_id] = paper_info
                else:
                    # 保留原有的 llm_approved 和 venue_info
                    original_llm_approved = json_data[topic][paper_id].get('llm_approved')
                    original_venue_info = json_data[topic][paper_id].get('venue_info')
                    
                    json_data[topic][paper_id] = paper_info
                    
                    if original_llm_approved is not None:
                        json_data[topic][paper_id]['llm_approved'] = original_llm_approved
                    if original_venue_info is not None:
                        json_data[topic][paper_id]['venue_info'] = original_venue_info
            
            if new_papers_count > 0:
                cnt[topic] = cnt.get(topic, 0) + new_papers_count
    
    for topic in json_data:
        if json_data[topic]:
            sorted_papers = sorted(
                json_data[topic].items(),
                key=lambda x: datetime.fromisoformat(
                    x[1]['published'].replace('Z', '+00:00')),
                reverse=True  # 最新的在前
            )
            json_data[topic] = dict(sorted_papers)
    
    with open(ARXIV_QUERY_CONFIG, 'r', encoding='utf-8') as f:
        config = json.load(f)
    config_order = list(config.keys())
    ordered_json_data = {}
    for topic in config_order:
        if topic in json_data:
            ordered_json_data[topic] = json_data[topic]
    
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(ordered_json_data, f, ensure_ascii=False, indent=2)

def get_current_paper_count(json_file, topic):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return len(data.get(topic, {}))
    except (FileNotFoundError, json.JSONDecodeError):
        return 0


if __name__ == "__main__":
    data_collector = []
    cnt = {}

    with open(ARXIV_QUERY_CONFIG, 'r', encoding='utf-8') as f:
        arxiv_filter_config = json.load(f)

    for topic, config in arxiv_filter_config.items():
        query = config['query']
        current_count = get_current_paper_count(ARXIV_PAPERS_JSON, topic)
        max_results = current_count + 10
        if 'max_results' in config:
            max_results = config['max_results']
        print(f"Querying {topic} with {query}")
        print(f"Current papers: {current_count}, Max results: {max_results}")
        query = query.replace("'", '"')
        data = get_daily_papers(topic, query, max_results, arxiv_filter_config, ARXIV_PAPERS_JSON)
        data_collector.append(data)

    md_file = "README.md"
    update_json_file(ARXIV_PAPERS_JSON, data_collector, cnt)
    with open(UPDATE_LOG, "r", encoding='utf-8') as f:
        origin_log = json.load(f)
    with open(UPDATE_LOG, "w", encoding='utf-8') as f:
        origin_log[datetime.now(timezone(timedelta(hours=8))).strftime(
            "%Y-%m-%d %H:%M:%S")] = cnt
        json.dump({key: value for key, value in sorted(
            origin_log.items(), reverse=True)}, f, indent=4)