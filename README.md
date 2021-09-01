# ASOUL-Generator-Backend

本项目为 https://asoul.infedg.xyz/ 的后端。
模型为基于 [CPM-Distill](https://github.com/TsinghuaAI/CPM-1-Distill) 的 [transformers](https://github.com/huggingface/transformers) 转化版本 [CPM-Generate-distill](https://huggingface.co/mymusise/CPM-Generate-distill/tree/main) 训练而成。
训练数据集：

- [asoul.icu](http://asoul.icu/)
- [枝江作文展](https://asoulcnki.asia/rank)

## 运行方式

#### 下载模型

[下载链接](https://disk.pku.edu.cn/#/link/88F0D3C9839329210503C7E50634AAFE)

需要将文件夹内的两个文件(`pytorch_model.bin` 和 `config.json`) 放入 `asoul_cpm` 文件夹下。

模型会不定期更新。

#### 安装依赖
```bash
pip install -r requirements.txt
```

#### 运行后端

```bash
python3 api.py
```

此时后端运行在 `5089` 端口。



