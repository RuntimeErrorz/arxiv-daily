#!/usr/bin/env python3
"""
独立的 JSON 转 Markdown 工具

从 arxiv_papers.json 生成 README.md 文件，支持venue信息显示。
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta


def clean_text(text):
    """清理文本，移除多余空格"""
    return ' '.join(text.split()).strip()


def json_to_md(json_filename, md_filename, config_filename):
    """
    将 arxiv_papers.json 转换为 Markdown 格式
    
    Args:
        json_filename: arxiv_papers.json 文件路径
        md_filename: 输出的 markdown 文件路径  
        config_filename: arxiv_query_config.json 文件路径
    """
    # 读取配置文件
    try:
        with open(config_filename, 'r', encoding='utf-8') as f:
            arxiv_query_config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ 无法读取配置文件 {config_filename}: {e}")
        sys.exit(1)
    
    # 读取论文数据
    try:
        with open(json_filename, "r", encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                data = {}
            else:
                data = json.loads(content)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ 无法读取数据文件 {json_filename}: {e}")
        sys.exit(1)

    # 生成时间戳
    time_now = str(datetime.now(
        timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S"))

    # 写入 Markdown 文件
    try:
        with open(md_filename, "w", encoding='utf-8') as f:
            f.write(f"## Updated at {time_now}\n\n")
            
            for topic, papers in data.items():
                topic_config = arxiv_query_config.get(topic, {})
                topic_prompt = topic_config.get('prompt', '')
                topic_query = topic_config.get('query', 'N/A')
                
                f.write(f"## {topic}\n\n")
                f.write(f'Query: {topic_query}\n\n')
                if topic_prompt:
                    f.write(f"Prompt: {topic_prompt}\n\n")

                # 表头：始终显示venue列和category列
                f.write("|Date|Title|Venue|Comments|Category|Journal|Authors|\n" +
                       "|---|---|---|---|---|---|---|\n")

                papers_written = 0
                for paper_id, paper_info in papers.items():
                    # 只显示通过AI筛选的论文（或没有设置筛选的论文）
                    if not paper_info.get('llm_approved', True):
                        continue
                        
                    authors = paper_info['authors']
                    published = datetime.fromisoformat(
                        paper_info['published'].replace('Z', '+00:00')
                    ).strftime('%Y-%m-%d')
                    arxiv_link = f'https://arxiv.org/abs/{paper_id}'
                    
                    # 获取venue信息
                    venue_info = paper_info.get('venue_info', {})
                    venue = venue_info.get('venue') or 'N/A'
                    
                    # 获取category信息
                    category = paper_info.get('primary_category', 'N/A')
                    
                    cleaned_data = {
                        'title': clean_text(paper_info['title']),
                        'journal': clean_text(paper_info.get('journal_ref', 'None') or 'None'),
                        'comment': clean_text(paper_info.get('comment', 'None') or 'None'),
                        'venue': clean_text(str(venue)),
                        'category': clean_text(str(category)),
                        'author_str': clean_text(f"{authors[0]} et al." if len(authors) > 2 else ', '.join(authors))
                    }

                    # 始终显示venue信息和category信息
                    f.write(
                        f"|**{published}**|**[{cleaned_data['title']}]({arxiv_link})**|"
                        f"{cleaned_data['venue']}|{cleaned_data['comment']}|{cleaned_data['category']}|{cleaned_data['journal']}|{cleaned_data['author_str']}|\n")
                    papers_written += 1

                if papers_written == 0:
                    f.write("|No relevant papers found||||||||\n")
                        
                f.write(f"\n")
                
        print(f"✅ 成功生成 {md_filename}")
        
    except Exception as e:
        print(f"❌ 写入文件失败: {e}")
        sys.exit(1)


def get_stats(json_filename):
    """获取统计信息"""
    try:
        with open(json_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        stats = {}
        total_papers = 0
        total_approved = 0
        total_with_venue = 0
        
        for topic, papers in data.items():
            topic_total = len(papers)
            topic_approved = sum(1 for p in papers.values() if p.get('llm_approved', True))
            topic_with_venue = sum(1 for p in papers.values() if p.get('venue_info'))
            
            stats[topic] = {
                'total': topic_total,
                'approved': topic_approved,
                'with_venue': topic_with_venue
            }
            
            total_papers += topic_total
            total_approved += topic_approved
            total_with_venue += topic_with_venue
        
        stats['_total'] = {
            'total': total_papers,
            'approved': total_approved,
            'with_venue': total_with_venue
        }
        
        return stats
        
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ 无法读取数据文件: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="将 arxiv_papers.json 转换为 Markdown 格式的 README.md",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python json_to_md.py                           # 基本转换（含venue）
  python json_to_md.py --stats                   # 仅显示统计信息
  python json_to_md.py --output custom.md        # 自定义输出文件
        """
    )
    
    parser.add_argument('--json', type=str, default='arxiv_papers.json',
                       help='输入的 JSON 文件路径 (默认: arxiv_papers.json)')
    parser.add_argument('--config', type=str, default='arxiv_query_config.json',
                       help='配置文件路径 (默认: arxiv_query_config.json)')
    parser.add_argument('--output', '-o', type=str, default='README.md',
                       help='输出的 Markdown 文件路径 (默认: README.md)')
    parser.add_argument('--stats', action='store_true',
                       help='仅显示统计信息，不生成文件')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.json):
        print(f"❌ 输入文件不存在: {args.json}")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        sys.exit(1)
    
    # 显示统计信息
    if args.stats:
        stats = get_stats(args.json)
        if stats:
            print("📊 论文统计信息:")
            print("=" * 60)
            for topic, info in stats.items():
                if topic == '_total':
                    print("-" * 60)
                    print(f"{'总计':<20} | {info['total']:>6} | {info['approved']:>8} | {info['with_venue']:>8}")
                else:
                    print(f"{topic:<20} | {info['total']:>6} | {info['approved']:>8} | {info['with_venue']:>8}")
            print("=" * 60)
            print(f"{'主题':<20} | {'总数':>6} | {'已审核':>8} | {'有venue':>8}")
        return
    
    # 执行转换
    print(f"🔄 开始转换...")
    print(f"   输入文件: {args.json}")
    print(f"   配置文件: {args.config}")
    print(f"   输出文件: {args.output}")
    
    json_to_md(args.json, args.output, args.config)
    
    # 显示简要统计
    stats = get_stats(args.json)
    if stats and '_total' in stats:
        total_stats = stats['_total']
        print(f"📊 共处理 {len(stats)-1} 个主题，{total_stats['approved']}/{total_stats['total']} 篇论文")
        if total_stats['with_venue'] > 0:
            print(f"   其中 {total_stats['with_venue']} 篇包含venue信息")


if __name__ == '__main__':
    main()
