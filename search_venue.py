import argparse
import json
import os
import sys
import time
from typing import Dict, Any, Optional

import requests
from tqdm import tqdm

ARXIV_PAPERS_JSON = "arxiv_papers.json"


def get_pub_info(arxiv_id: str = None,
                 timeout=15,
                 proxies=None
):
    """通过 Semantic Scholar 获取 venue 等信息。
    返回示例：{"source":"semanticscholar","venue":"CVPR 2025","type":"Conference","publisher":null}
    """

    url = (
        "https://api.semanticscholar.org/graph/v1/paper/"
        f"arXiv:{arxiv_id}?fields=title,venue,publicationTypes,publicationVenue,year,externalIds"
    )

    print(url)

    attempt = 0
    while True:
        attempt += 1
        try:
            r = requests.get(url, timeout=timeout, proxies=proxies)
            
            if r.status_code == 429:
                print(f"   ⏳ 遇到错误({r.status_code})，立即重试 (第{attempt}次)...")
                continue
            
            r.raise_for_status()
            data = r.json()
            
            # 打印原始响应用于调试
            print(f"🔍 [{arxiv_id}] Semantic Scholar 响应 (第{attempt}次尝试):")
            print(f"   URL: {url}")
            print(f"   响应数据: {json.dumps(data, ensure_ascii=False, indent=2)[:500]}...")
            
            paper = data
            if not paper:
                print(f"   ❌ 未找到论文数据")
                return {}
            
            pv = paper.get("publicationVenue") or {}
            venue = paper.get("venue") or pv.get("name")
            pub_types = paper.get("publicationTypes") or []
            
            print(f"   📝 解析结果: venue='{venue}', types={pub_types}")
            
            return {
                "source": "semanticscholar",
                "venue": venue,
                "type": ",".join(pub_types) or None,
                "publisher": None,  # Semantic Scholar 通常不提供 publisher
            }
            
        except requests.exceptions.RequestException as e:
            if e.response is not None and e.response.status_code == 404:
                break
            print(f"   ⚠️  请求异常: {e}，立即重试 (第{attempt}次)...")
            continue


def update_topic_venues(topic: str,
                        json_path: str = ARXIV_PAPERS_JSON,
                        only_missing: bool = True,
                        half_mode: bool = False,
                        max_items: int = 0,
                        timeout: int = 15,
                        proxy: Optional[str] = None) -> Dict[str, Any]:
    """为指定 topic 的论文批量补齐 venue_info 信息。
    返回统计信息。
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(json_path)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if topic not in data:
        raise KeyError(f"topic 不存在: {topic}")

    topic_papers: Dict[str, Dict[str, Any]] = data.get(topic, {})

    proxies = None
    if proxy:
        # 支持不带协议的写法 127.0.0.1:7890
        if '://' not in proxy:
            proxy = f"http://{proxy}"
        proxies = {"http": proxy, "https": proxy}

    updated = 0
    skipped = 0
    failed = 0
    touched_ids = []

    # 仅处理 llm_approved 为 True 的条目（严格 True，缺失或 False 均不处理）
    all_items = list(topic_papers.items())
    pre_filtered_items = [(pid, inf) for pid, inf in all_items if inf.get('llm_approved') is True]
    processed_items = pre_filtered_items
    if max_items and max_items > 0:
        processed_items = processed_items[:max_items]

    for paper_id, info in tqdm(processed_items, desc="处理论文", unit="篇"):
        # 根据模式决定是否处理
        if only_missing:
            # 仅在缺失 venue_info 时处理
            if info.get('venue_info'):
                skipped += 1
                continue
        elif half_mode:
            # half模式：处理缺失venue_info或venue为null的
            venue_info = info.get('venue_info')
            if venue_info and venue_info.get('venue') is not None:
                skipped += 1
                continue
        # 如果是 --all 模式，不跳过任何条目
        try:
            res = get_pub_info(
                arxiv_id=paper_id,
                timeout=timeout,
                proxies=proxies,
            )
        except Exception:
            failed += 1
            continue

        # 如果 API 请求失败，res 会是空字典，不处理
        if not res:
            failed += 1
            continue

        # 合并更新：将新字段放入 venue_info 字典，不动原有的 doi/journal_ref
        before = json.dumps(info.get('venue_info', {}), ensure_ascii=False)

        venue_info = {
            'venue_source': 'Semantic Scholar',
            'queried_at': time.strftime('%Y-%m-%d')  # 记录查询日期
        }
        
        venue = res.get('venue')
        if venue:
            venue_info['venue'] = venue
            venue_info['pub_type'] = res.get('type')
            venue_info['publisher'] = res.get('publisher')
            print(f"   ✅ 更新成功: {venue}")
        else:
            # 只有在成功请求但没有找到venue时，才设置占位信息
            venue_info['venue'] = None
            venue_info['pub_type'] = res.get('type')
            venue_info['publisher'] = None
            venue_info['note'] = '已查询但未找到venue信息'
            print(f"   📝 已标记为查询过（无venue信息）")
        
        # 总是更新 venue_info（成功请求的情况下）
        info['venue_info'] = venue_info

        after = json.dumps(info.get('venue_info', {}), ensure_ascii=False)
        if before != after:
            updated += 1
            touched_ids.append(paper_id)
        else:
            skipped += 1

    data[topic] = topic_papers

    return {
        'updated': updated,
        'skipped': skipped,
        'failed': failed,
        'touched_ids': touched_ids,
        'data': data,
        # 统计信息：预筛选与实际处理数
        'total_in_topic': len(all_items),
        'pre_filtered_count': len(pre_filtered_items),
        'processed_count': len(processed_items),
    }


def main():
    parser = argparse.ArgumentParser(description="为指定 topic 的 arxiv_papers.json 补充 venue_info 信息")
    parser.add_argument('--topic', type=str, required=True, help='要处理的 topic 名称（一级键）')
    parser.add_argument('--json', type=str, default=ARXIV_PAPERS_JSON, help='arxiv_papers.json 路径')
    parser.add_argument('--only-missing', action='store_true', help='仅在缺失 venue_info 时更新（默认行为）')
    parser.add_argument('--half', action='store_true', help='处理缺失 venue_info 或 venue 为 null 的论文')
    parser.add_argument('--all', dest='only_missing', action='store_false', help='对所有条目尝试更新（覆盖模式）')
    parser.add_argument('--max', type=int, default=0, help='最多处理多少条，0 表示不限制')
    parser.add_argument('--timeout', type=int, default=15, help='HTTP 请求超时时间（秒）')
    parser.add_argument('--proxy', type=str, default=None, help='代理地址，如 http://127.0.0.1:7890 或 127.0.0.1:7890')
    parser.add_argument('--use-proxy-7890', action='store_true', help='快捷开关：等效于 --proxy http://127.0.0.1:7890')
    parser.add_argument('--dry-run', action='store_true', help='试运行，不写回文件')

    args = parser.parse_args()

    # 模式互斥检查
    mode_count = sum([args.only_missing, args.half, not args.only_missing])
    if mode_count > 1:
        print("❌ 错误：--only-missing, --half, --all 三个模式只能选择一个")
        sys.exit(1)

    proxy = args.proxy
    if args.use_proxy_7890 and not proxy:
        proxy = 'http://127.0.0.1:7890'

    try:
        result = update_topic_venues(
            topic=args.topic,
            json_path=args.json,
            only_missing=args.only_missing,
            half_mode=args.half,
            max_items=args.max,
            timeout=args.timeout,
            proxy=proxy,
        )
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        sys.exit(1)

    # 预筛选统计输出
    print(
        f"🧮 筛选 llm_approved=True: {result['pre_filtered_count']}/{result['total_in_topic']}；实际处理: {result['processed_count']}"
    )
    print(f"📊 处理完成: updated={result['updated']}, skipped={result['skipped']}, failed={result['failed']}")
    if result['touched_ids']:
        print("   变更的 paper_id 示例:", ', '.join(result['touched_ids'][:10]))

    if args.dry_run:
        print("🔎 dry-run：未写回文件")
        return

    # 写回
    try:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(result['data'], f, ensure_ascii=False, indent=2)
        print(f"✅ 已写回 {args.json}")
    except Exception as e:
        print(f"❌ 写回失败: {e}")
        sys.exit(2)


if __name__ == '__main__':
    main()
