import pandas as pd
from transformers import DistilBertTokenizer
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json
import argparse
import os

# 初始化分词器（全局变量）
tokenizer = DistilBertTokenizer.from_pretrained("./models/distilbert-base-uncased")


# 多进程分词函数
def tokenize_url(url):
    return tokenizer.tokenize(url)


def tokenize_and_count_frequencies(urls):
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(tokenize_url, urls), desc="Tokenizing URLs", total=len(urls)))
    all_tokens = [token for tokens in results for token in tokens]
    token_frequencies = Counter(all_tokens)
    return token_frequencies


# 构建倒排索引函数，增加进度条
def build_inverted_index(urls):
    inverted_index = defaultdict(list)
    for idx, url in tqdm(enumerate(urls), total=len(urls), desc="Building Inverted Index"):
        tokens = set(tokenizer.tokenize(url))  # 使用集合避免重复的词
        for token in tokens:
            inverted_index[token].append(idx)
    return inverted_index


# 提取稀有词的上下文（顶级函数，支持多进程）
def context_worker(args):
    word, inverted_index, urls, max_contexts = args
    contexts = []
    url_indices = inverted_index.get(word, [])
    for idx in url_indices:
        contexts.append(urls[idx])
        if len(contexts) >= max_contexts:  # 限制返回的上下文数量
            break
    return word, contexts


def generate_words_with_contexts(rare_words, urls, inverted_index, max_contexts):
    # 将 rare_words 和 urls 打包为参数列表
    task_args = [(word, inverted_index, urls, max_contexts) for word in rare_words]
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(context_worker, task_args),
                            desc="Generating Rare Word Contexts",
                            total=len(rare_words)))
    words_with_contexts = {word: contexts for word, contexts in results}
    return words_with_contexts


# 主函数
def main(train_file, threshold_percent, max_contexts, output_dir):
    print("Loading data...")
    urls = pd.read_csv(train_file)["url"].tolist()

    print("Tokenizing URLs and counting token frequencies...")
    token_frequencies = tokenize_and_count_frequencies(urls)

    print(f"Extracting rare words with threshold {threshold_percent}%...")
    sorted_frequencies = sorted(token_frequencies.values())
    threshold = sorted_frequencies[int(len(sorted_frequencies) * threshold_percent // 100)]
    rare_words = [word for word, freq in token_frequencies.items() if freq <= threshold]

    print(f"Generating inverted index...")
    inverted_index = build_inverted_index(urls)

    print(f"Generating rare word contexts with max {max_contexts} contexts per word...")
    words_with_contexts = generate_words_with_contexts(rare_words, urls, inverted_index, max_contexts)

    # 格式化输出文件名
    output_file_name = f"rare_words_with_contexts_test0.1_th={threshold_percent}_maxctx={max_contexts}.json"
    output_file_path = os.path.join(output_dir, output_file_name)

    print(f"Saving rare word dictionary to {output_file_path}...")
    os.makedirs(output_dir, exist_ok=True)  # 如果输出目录不存在则创建
    with open(output_file_path, "w") as f:
        json.dump(words_with_contexts, f, indent=4)
    print("Done!")


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Generate rare word contexts from URLs.")
    parser.add_argument('--train_file', type=str, default='data/Test_Split_0.1.csv', help="Path to the training CSV file.")
    parser.add_argument('--threshold_percent', type=int, default=20,
                        help="Threshold percent for rare word extraction.")
    parser.add_argument('--max_contexts', type=int, default=5,
                        help="Maximum number of contexts to return per rare word.")
    parser.add_argument('--output_dir', type=str, default='data/', help="Directory to save the output JSON file.")

    return parser.parse_args()


# 执行代码
if __name__ == "__main__":
    args = parse_args()
    main(args.train_file, args.threshold_percent, args.max_contexts, args.output_dir)
