# vgcn-bertram

## 目录结构

- models 存放模型文件，需要下载vgcn-bert-distilbert-base-uncased，bertram-add-for-bert-base-uncased https://www.cis.uni-muenchen.de/~schickt/bertram-add-for-bert-base-uncased.zip
- data 存放数据集

## 运行

### 生成稀有词

```bash
python rare_word_generator.py --train_file data/Test_Split_0.1.csv --threshold_percent 20 --max_contexts 5 --output_dir data/

python process_json.py
```



### 训练

```bash
python new-seqclass+train+bertram.py --json_path data/rare_words.json
```



### 评估

```bash
python eval+bertram.py
```



## 参考

vgcnbert: https://huggingface.co/zhibinlu/vgcn-bert-distilbert-base-uncased

bertram: https://github.com/timoschick/bertram



## Note

vgcnbert 的源码当时不能直接运行，有bug，我已经做了修改，所以不要动 `configuration_vgcn_bert.py`