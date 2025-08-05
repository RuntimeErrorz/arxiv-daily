import json
import os
import time
import google.generativeai as genai
from tqdm import tqdm

ARXIV_PAPERS_JSON = "arxiv_papers.json"
ARXIV_QUERY_CONFIG = "arxiv_query_config.json"
MODEL = "gemini-2.5-flash-lite"


def setup_gemini():
    """初始化Gemini模型"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("错误: 未找到GEMINI_API_KEY环境变量")
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
    """使用Gemini评估论文是否符合标准"""
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


def refilter_topic(topic, papers, config, gemini_model):
    """重新筛选某个topic下的所有论文"""
    topic_prompt = config.get(topic, {}).get('prompt', '')
    
    if not topic_prompt:
        print(f"⚠️  {topic} 没有设置筛选标准，跳过")
        return papers, 0, 0, []
    
    if not gemini_model:
        print(f"⚠️  Gemini模型未初始化，跳过 {topic}")
        return papers, 0, 0, []
    
    print(f"\n🔄 开始重新筛选 {topic} ({len(papers)} 篇论文)")
    print(f"筛选标准: {topic_prompt[:100]}...")
    
    approved_count = 0
    rejected_count = 0
    inconsistent_papers = []  # 记录前后不一致的论文
    
    for paper_id, paper_info in tqdm(papers.items(), desc=f"重新筛选 {topic}"):
        # 记录原始筛选结果
        original_approved = paper_info.get('llm_approved', True)  # 默认为True（未筛选过的论文）
        
        # 使用AI重新评估
        is_relevant = evaluate_paper_with_gemini(
            gemini_model,
            paper_info['title'],
            paper_info['summary'],
            topic_prompt
        )
        
        # 检查前后是否一致
        if original_approved != is_relevant:
            inconsistent_info = {
                'paper_id': paper_id,
                'title': paper_info['title'],
                'original_result': original_approved,
                'new_result': is_relevant,
                'change_type': '通过→拒绝' if original_approved and not is_relevant else '拒绝→通过'
            }
            inconsistent_papers.append(inconsistent_info)
            print(f"🔄 {paper_id}: {inconsistent_info['change_type']} - {paper_info['title'][:60]}...")
        
        # 更新筛选结果
        paper_info['llm_approved'] = is_relevant
        
        if is_relevant:
            approved_count += 1
            if original_approved == is_relevant:  # 结果一致
                print(f"✅ {paper_id}: {paper_info['title'][:60]}...")
        else:
            rejected_count += 1
            if original_approved == is_relevant:  # 结果一致
                print(f"❌ {paper_id}: {paper_info['title'][:60]}...")
        
            
    print(f"✅ {topic} 重新筛选完成:")
    print(f"   - 通过: {approved_count} 篇")
    print(f"   - 拒绝: {rejected_count} 篇") 
    print(f"   - 结果变化: {len(inconsistent_papers)} 篇")
    
    return papers, approved_count, rejected_count, inconsistent_papers


def update_refiltering_status(config_file, topics_processed):
    """将已处理的topic的refiltering状态置为false"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        updated = False
        for topic in topics_processed:
            if topic in config and config[topic].get('refiltering', False):
                config[topic]['refiltering'] = False
                updated = True
                print(f"📝 已将 {topic} 的refiltering状态置为false")
        
        if updated:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            print("✅ 配置文件已更新")
        
        return updated
    except Exception as e:
        print(f"❌ 更新配置文件失败: {e}")
        return False


def save_inconsistent_results(inconsistent_data, filename="refilter_changes.json"):
    """保存筛选结果变化的论文信息"""
    try:
        # 准备保存的数据
        save_data = {
            'refilter_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_changes': sum(len(changes) for changes in inconsistent_data.values()),
            'topics': inconsistent_data
        }
        
        # 保存到JSON文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"📄 筛选结果变化已保存到 {filename}")
        return True
    except Exception as e:
        print(f"❌ 保存变化记录失败: {e}")
        return False


def print_inconsistent_summary(inconsistent_data):
    """打印筛选结果变化的汇总信息"""
    total_changes = sum(len(changes) for changes in inconsistent_data.values())
    
    if total_changes == 0:
        print("\n🎉 所有论文的筛选结果都保持一致，没有变化！")
        return
    
    print(f"\n📊 筛选结果变化汇总 (共 {total_changes} 篇论文):")
    print("=" * 60)
    
    for topic, changes in inconsistent_data.items():
        if not changes:
            continue
            
        print(f"\n📂 {topic} ({len(changes)} 篇变化):")
        
        # 统计变化类型
        pass_to_reject = sum(1 for c in changes if c['change_type'] == '通过→拒绝')
        reject_to_pass = sum(1 for c in changes if c['change_type'] == '拒绝→通过')
        
        print(f"   📈 拒绝→通过: {reject_to_pass} 篇")
        print(f"   📉 通过→拒绝: {pass_to_reject} 篇")
        
        # 显示前几篇变化的论文
        print("   📋 主要变化:")
        for i, change in enumerate(changes[:5]):  # 只显示前5篇
            status_icon = "📈" if change['change_type'] == '拒绝→通过' else "📉"
            print(f"      {status_icon} {change['paper_id']}: {change['title'][:50]}...")
        
        if len(changes) > 5:
            print(f"      ... 还有 {len(changes) - 5} 篇论文发生变化")


def main():
    """主函数"""
    print("🚀 开始执行论文重新筛选...")
    
    # 读取配置文件
    try:
        with open(ARXIV_QUERY_CONFIG, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ 配置文件 {ARXIV_QUERY_CONFIG} 不存在")
        return
    except json.JSONDecodeError:
        print(f"❌ 配置文件 {ARXIV_QUERY_CONFIG} 格式错误")
        return
    
    # 读取论文数据
    try:
        with open(ARXIV_PAPERS_JSON, 'r', encoding='utf-8') as f:
            papers_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 论文数据文件 {ARXIV_PAPERS_JSON} 不存在")
        return
    except json.JSONDecodeError:
        print(f"❌ 论文数据文件 {ARXIV_PAPERS_JSON} 格式错误")
        return
    
    # 初始化Gemini模型
    gemini_model = setup_gemini()
    if not gemini_model:
        print("❌ 无法初始化Gemini模型，请检查GEMINI_API_KEY环境变量")
        return
    
    # 找到需要重新筛选的topics
    topics_to_refilter = []
    for topic, topic_config in config.items():
        if topic_config.get('refiltering', False) and topic_config.get('prompt', ''):
            topics_to_refilter.append(topic)
    
    if not topics_to_refilter:
        print("ℹ️  没有找到需要重新筛选的topic")
        print("请在配置文件中设置 'refiltering': true 和 'prompt' 字段")
        return
    
    print(f"📋 找到 {len(topics_to_refilter)} 个需要重新筛选的topic: {', '.join(topics_to_refilter)}")
    
    # 统计信息
    total_approved = 0
    total_rejected = 0
    processed_topics = []
    all_inconsistent_data = {}  # 记录所有topic的变化数据
    
    # 对每个topic进行重新筛选
    for topic in topics_to_refilter:
        if topic not in papers_data:
            print(f"⚠️  {topic} 在论文数据中不存在，跳过")
            continue
        
        topic_papers = papers_data[topic]
        if not topic_papers:
            print(f"⚠️  {topic} 没有论文数据，跳过")
            continue
        
        # 重新筛选
        updated_papers, approved, rejected, inconsistent_papers = refilter_topic(
            topic, topic_papers, config, gemini_model
        )
        
        # 更新数据
        papers_data[topic] = updated_papers
        total_approved += approved
        total_rejected += rejected
        processed_topics.append(topic)
        all_inconsistent_data[topic] = inconsistent_papers
    
    if processed_topics:
        # 保存更新后的论文数据
        try:
            with open(ARXIV_PAPERS_JSON, 'w', encoding='utf-8') as f:
                json.dump(papers_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 论文数据已保存到 {ARXIV_PAPERS_JSON}")
        except Exception as e:
            print(f"❌ 保存论文数据失败: {e}")
            return
        
        # 更新配置文件，将refiltering置为false
        config_updated = update_refiltering_status(ARXIV_QUERY_CONFIG, processed_topics)
        
        # 打印和保存变化汇总
        print_inconsistent_summary(all_inconsistent_data)
        save_inconsistent_results(all_inconsistent_data)
        
        # 输出统计信息
        print(f"\n📊 重新筛选完成统计:")
        print(f"   - 处理的topics: {len(processed_topics)}")
        print(f"   - 通过筛选: {total_approved} 篇")
        print(f"   - 被拒绝: {total_rejected} 篇")
        print(f"   - 总计处理: {total_approved + total_rejected} 篇")
        print(f"   - 结果变化: {sum(len(changes) for changes in all_inconsistent_data.values())} 篇")
        
        if config_updated:
            print(f"\n✅ 重新筛选任务完成，配置文件已自动更新")
        else:
            print(f"\n⚠️  重新筛选任务完成，但配置文件更新失败")
    else:
        print("❌ 没有成功处理任何topic")


if __name__ == "__main__":
    main()
