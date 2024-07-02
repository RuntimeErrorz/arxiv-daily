import arxiv
import json
import requests

from datetime import datetime, timezone, timedelta
from tqdm import tqdm

BASE_URL = "https://arxiv.paperswithcode.com/api/v0/papers/"
cnt = {}


def get_daily_papers(topic, query, max_results):
    content = dict()
    res = arxiv.Client(delay_seconds=3, num_retries=3).results(
        arxiv.Search(
            query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate
        )
    )
    for result in tqdm(res):
        if result.primary_category != 'cs.CV':
            continue
        paper_id = result.get_short_id()
        comment = result.comment.replace(
            "\n", " ") if result.comment else "None"
        journal_ref = result.journal_ref.replace(
            "\n", " ") if result.journal_ref else "None"
        # eg: 2108.09112v1 -> 2108.09112
        ver_pos = paper_id.find('v')
        if ver_pos == -1:
            paper_key = paper_id
        else:
            paper_key = paper_id[0:ver_pos]
        try:
            r = requests.get(BASE_URL + paper_id).json()
            prefix = f"|**{result.published.date()}**|**[{result.title}]({result.entry_id[:-2]})**|{journal_ref}"
            suffix = f"{comment}|{result.authors[0]} et.al.|\n"
            content[paper_key] = prefix + f"|**[link]({r['official']['url']})**|" + \
                suffix if "official" in r and r["official"] else prefix + \
                f"|None|" + suffix
        except Exception as e:
            print(f"exception: {e} with id: {paper_key}")
    data = {topic: content}
    return data


def update_json_file(filename, data_all):
    with open(filename, "r", encoding='utf-8') as f:
        content = f.read()
        if not content:
            m = {}
        else:
            m = json.loads(content)
    json_data = m.copy()
    for data in data_all:
        for query_class in data.keys():
            papers = data[query_class]
            if query_class in json_data.keys():  # MVS, Depth
                for key in papers.keys():  # 2108.09112
                    if key not in json_data[query_class].keys():
                        if query_class in cnt.keys():
                            cnt[query_class] += 1
                        else:
                            cnt[query_class] = 1
                json_data[query_class].update(papers)
            else:
                json_data[query_class] = papers
                for key in papers.keys():  # 2108.09112
                    if query_class in cnt.keys():
                        cnt[query_class] += 1
                    else:
                        cnt[query_class] = 1
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(json_data, f)


def json_to_md(filename, md_filename):
    time_now = str(datetime.now(
        timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S"))
    with open(filename, "r", encoding='utf-8') as f:
        content = f.read()
        if not content:
            data = {}
        else:
            data = json.loads(content)
    with open(md_filename, "w", encoding='utf-8') as f:
        f.write(f"## Updated on {time_now}\n\n")
        for keyword in data.keys():
            day_content = data[keyword]
            if not day_content:
                continue
            f.write(f"## {keyword}\n\n")
            f.write("|Published Date|Title|Journal|Code|Comments|Authors\n" +
                    "|---|---|---|---|---|---|\n")
            day_content = {key: value for key, value in sorted(
                day_content.items(), reverse=True)}
            for _, v in day_content.items():
                if v is not None:
                    f.write(v)
            f.write(f"\n")


if __name__ == "__main__":
    data_collector = []
    with open('arxiv_query_config.json', 'r', encoding='utf-8') as f:
        arxiv_filter_config = json.load(f)
    for topic, query in arxiv_filter_config.items():
        print(f"Querying {topic} with {query}")
        query = query.replace("'", '"')
        data = get_daily_papers(topic, query, max_results=10)
        data_collector.append(data)
    json_file = "cv_arxiv_daily.json"
    md_file = "README.md"
    update_json_file(json_file, data_collector)
    json_to_md(json_file, md_file)
    cnt_log = "daily_update_log.json"
    with open(cnt_log, "r", encoding='utf-8') as f:
        origin_log = json.load(f)
    with open(cnt_log, "w", encoding='utf-8') as f:
        origin_log[datetime.now(timezone(timedelta(hours=8))).strftime(
            "%Y-%m-%d %H:%M:%S")] = cnt
        json.dump({key: value for key, value in sorted(
            origin_log.items(), reverse=True)}, f, indent=4)
