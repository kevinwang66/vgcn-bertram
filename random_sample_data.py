import pandas as pd
import random
import os
import argparse


# 计算并显示 URL 标签占比
def display_label_distribution(df, file_name):
    label_counts = df['label'].value_counts(normalize=True) * 100
    print(f"URL 标签占比 - {file_name}:")
    for label, percent in label_counts.items():
        print(f"标签 {label}: {percent:.2f}%")
    print()


# 随机选取 50% 数据并保存为新的 CSV 文件
def process_and_save_csv(input_file, output_dir, output_file_name):
    # 读取 CSV 文件
    df = pd.read_csv(input_file)

    # 随机选取 50% 数据
    sampled_df = df.sample(frac=0.5, random_state=42)  # 设置随机种子确保可重复性

    # 显示标签占比
    display_label_distribution(df, input_file)

    # 保存为新的 CSV 文件
    os.makedirs(output_dir, exist_ok=True)  # 如果目录不存在则创建
    output_file_path = os.path.join(output_dir, output_file_name)
    sampled_df.to_csv(output_file_path, index=False)
    print(f"已保存文件: {output_file_path}")

    return sampled_df


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly sample 50% data from Test.csv and Train.csv and save as new CSV files.")
    parser.add_argument('--train_file', type=str, default='Grambedding_dataset/Train.csv', help="Path to the Train.csv file.")
    parser.add_argument('--test_file', type=str, default='Grambedding_dataset/Test.csv', help="Path to the Test.csv file.")
    parser.add_argument('--output_dir', type=str, default='exp_bertram+vgcnbert_50%/data-50%r', help="Directory to save the output CSV files.")

    return parser.parse_args()


# 主函数
def main():
    # 解析命令行参数
    args = parse_args()

    # 处理 Train.csv 文件
    print(f"Processing {args.train_file}...")
    train_output_file_name = "Train_sampled_50.csv"
    sampled_train_df = process_and_save_csv(args.train_file, args.output_dir, train_output_file_name)

    # 处理 Test.csv 文件
    print(f"Processing {args.test_file}...")
    test_output_file_name = "Test_sampled_50.csv"
    sampled_test_df = process_and_save_csv(args.test_file, args.output_dir, test_output_file_name)


if __name__ == "__main__":
    main()
