import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from typing import Dict, Any, List, Tuple, Optional


ARXIV_PAPERS_JSON = "arxiv_papers.json"


def load_papers(json_path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(json_path):
        print(f"❌ 未找到数据文件: {json_path}")
        sys.exit(1)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ 解析 JSON 失败: {e}")
        sys.exit(1)


def sanitize_filename(name: str, replacement: str = "_") -> str:
    # Windows 禁止的字符：<>:"/\|?*
    name = re.sub(r'[<>:"/\\|?*]+', replacement, name)
    # 去除控制字符
    name = re.sub(r'[\x00-\x1f]', '', name)
    # 收尾空白
    name = name.strip()
    # 避免过长
    return name[:160] if len(name) > 160 else name


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_pdf_url(paper_id: str, raw_url: str = None) -> List[str]:
    candidates = []
    if raw_url:
        candidates.append(raw_url)
        # 有些链接没有 .pdf 后缀，作为备选再拼一个标准 pdf 链接
        if not raw_url.lower().endswith('.pdf'):
            candidates.append(f"https://arxiv.org/pdf/{paper_id}.pdf")
    else:
        candidates.append(f"https://arxiv.org/pdf/{paper_id}.pdf")
    # 去重保持顺序
    seen = set()
    ordered = []
    for u in candidates:
        if u not in seen:
            ordered.append(u)
            seen.add(u)
    return ordered


def download_with_retries(urls: List[str], dest_path: str, retries: int = 3, timeout: int = 60, opener: Optional[urllib.request.OpenerDirector] = None) -> Tuple[bool, str]:
    """尝试多个候选 URL 进行下载，失败重试。

    返回 (success, used_url)
    """
    ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/127.0 Safari/537.36")
    last_err = None
    for url in urls:
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': ua})
                if opener is not None:
                    resp_ctx = opener.open(req, timeout=timeout)
                else:
                    resp_ctx = urllib.request.urlopen(req, timeout=timeout)
                with resp_ctx as resp:
                    # 流式写入
                    with open(dest_path, 'wb') as out:
                        while True:
                            chunk = resp.read(1024 * 64)
                            if not chunk:
                                break
                            out.write(chunk)
                return True, url
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                last_err = e
                if attempt < retries:
                    time.sleep(min(2 * attempt, 5))
                else:
                    # 当前 URL 用尽重试，尝试下一个 URL
                    pass
        # 切换到下一个候选 URL
    if last_err:
        return False, f"最后错误: {last_err}"
    return False, "未知错误"


def pick_approved(papers: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    # 沿用项目默认：未设置 llm_approved 时视作 True
    items: List[Tuple[str, Dict[str, Any]]] = []
    for pid, info in papers.items():
        if info.get('llm_approved', True):
            items.append((pid, info))
    return items


def create_proxy_opener(proxy: str) -> urllib.request.OpenerDirector:
    """基于 HTTP/HTTPS 代理地址创建 opener。

    只支持 HTTP/HTTPS 代理（例如 http://127.0.0.1:7890）。
    """
    if not proxy:
        raise ValueError("empty proxy")
    # 若未带协议，默认 http
    if '://' not in proxy:
        proxy = f"http://{proxy}"
    proxies = {
        'http': proxy,
        'https': proxy,
    }
    handler = urllib.request.ProxyHandler(proxies)
    opener = urllib.request.build_opener(handler)
    return opener


def main():
    parser = argparse.ArgumentParser(description="按 topic 下载通过 llm_approved 的 arXiv 论文 PDF")
    parser.add_argument("--topic", type=str, help="Topic 名称（来自 arxiv_papers.json 的一级键）")
    parser.add_argument("--out-dir", type=str, default="downloads", help="下载根目录，默认 downloads")
    parser.add_argument("--list-topics", action="store_true", help="仅列出可用 topic 并退出")
    parser.add_argument("--dry-run", action="store_true", help="仅列出将要下载的论文，不实际下载")
    parser.add_argument("--max", type=int, default=0, help="最多下载多少篇（0 表示不限制）")
    parser.add_argument("--retries", type=int, default=3, help="失败重试次数，默认 3")
    parser.add_argument("--sleep", type=float, default=0.0, help="每次下载之间的间隔秒数，默认 0")
    parser.add_argument("--proxy", type=str, default=None, help="代理地址，如 http://127.0.0.1:7890（支持 http/https 代理）")
    parser.add_argument("--use-proxy-7890", action="store_true", help="快捷开关：等效于 --proxy http://127.0.0.1:7890")

    args = parser.parse_args()

    data = load_papers(ARXIV_PAPERS_JSON)

    if args.list_topics:
        print("可用 topics:")
        for t in data.keys():
            print(f" - {t}")
        return

    if not args.topic:
        print("❌ 未提供 --topic。可先用 --list-topics 查看可用项。")
        sys.exit(2)

    if args.topic not in data:
        # 尝试不区分大小写匹配
        lower_map = {k.lower(): k for k in data.keys()}
        if args.topic.lower() in lower_map:
            topic_key = lower_map[args.topic.lower()]
        else:
            print(f"❌ topic 不存在: {args.topic}")
            print("可用 topics:")
            for t in data.keys():
                print(f" - {t}")
            sys.exit(3)
    else:
        topic_key = args.topic

    topic_papers = data.get(topic_key, {})
    approved = pick_approved(topic_papers)

    if not approved:
        print(f"ℹ️  {topic_key} 下没有通过筛选 (llm_approved) 的论文。")
        return

    # 目的目录：out-dir/<topic>
    topic_dir = os.path.join(args.out_dir, sanitize_filename(topic_key, "_"))
    ensure_dir(topic_dir)

    # 统计
    to_download: List[Tuple[str, Dict[str, Any]]] = approved
    if args.max and args.max > 0:
        to_download = to_download[:args.max]

    print(f"📚 Topic: {topic_key}")
    print(f"✅ 通过筛选论文数: {len(approved)}；计划下载: {len(to_download)}；输出目录: {topic_dir}")

    if args.dry_run:
        for pid, info in to_download:
            title = info.get('title', '')
            print(f" - {pid}: {title[:100]}")
        print("（dry-run 结束，未进行实际下载）")
        return

    downloaded = 0
    skipped = 0
    failed: List[Tuple[str, str]] = []

    # 代理设置
    opener: Optional[urllib.request.OpenerDirector] = None
    proxy_cfg = args.proxy
    if args.use_proxy_7890 and not proxy_cfg:
        proxy_cfg = "http://127.0.0.1:7890"
    if proxy_cfg:
        try:
            opener = create_proxy_opener(proxy_cfg)
            print(f"🌐 使用代理: {proxy_cfg}")
        except Exception as e:
            print(f"⚠️  代理无效，将不使用代理。原因: {e}")

    for idx, (pid, info) in enumerate(to_download, start=1):
        title = info.get('title', '') or pid
        fname = f"{pid} - {sanitize_filename(title)}.pdf"
        dest = os.path.join(topic_dir, fname)

        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            skipped += 1
            print(f"[{idx}/{len(to_download)}] ⏭️  已存在，跳过: {fname}")
            continue

        urls = build_pdf_url(pid, info.get('pdf_url'))
        print(f"[{idx}/{len(to_download)}] ⬇️  正在下载: {title[:80]}…")
        ok, used = download_with_retries(urls, dest, retries=max(1, args.retries), opener=opener)
        if ok:
            downloaded += 1
            print(f"    ✅ 完成 -> {os.path.basename(dest)}")

        else:
            failed.append((pid, str(used)))
            # 清理不完整文件
            try:
                if os.path.exists(dest) and os.path.getsize(dest) == 0:
                    os.remove(dest)
            except Exception:
                pass
            print(f"    ❌ 失败: {title[:80]}… | {used}")

    print("\n📊 总结：")
    print(f"   ✅ 成功: {downloaded}")
    print(f"   ⏭️  跳过: {skipped}")
    print(f"   ❌  失败: {len(failed)}")
    if failed:
        print("   失败列表（paper_id -> error/url）：")
        for pid, err in failed[:10]:  # 仅展示前 10 个
            print(f"   - {pid} -> {err}")
        if len(failed) > 10:
            print(f"   ... 还有 {len(failed) - 10} 条")


if __name__ == "__main__":
    main()
