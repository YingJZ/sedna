# LLM 联合推理服务示例

本例介绍如何在大语言模型（LLM）场景下，使用 Sedna 的联合推理服务。通过 edge（小模型）和 cloud（大模型）协同推理，实现推理效率和效果的平衡。

## 目录结构

```
examples/joint_inference/llm/
├── big_model/
│   ├── big_model.py
│   └── interface.py
├── little_model/
│   ├── little_model.py
│   └── interface.py
├── llm_inference.yaml
└── README.md
```

## 环境准备

1. 安装 Sedna，参考 [Sedna 安装文档](/docs/setup/install.md)
2. Edge 节点和 Cloud 节点需具备 Python3、transformers、torch 等依赖

## 模型准备

本例直接使用 Huggingface Hub 上的模型：
- edge 侧：distilgpt2
- cloud 侧：gpt2-xl

如需自定义模型，可将模型下载到本地，并在 yaml 配置中指定路径。

## 镜像准备

请基于官方 Sedna 镜像，增加 transformers/torch 依赖，并将 big_model、little_model 目录下的代码拷贝到镜像内。

## 创建联合推理服务

1. 修改 `llm_inference.yaml`，设置 `$EDGE_NODE` 和 `$CLOUD_NODE` 为实际节点名。
2. 创建输出目录（edge 节点）：
   ```
   mkdir -p /joint_inference/output
   ```
3. 部署服务：
   ```
   kubectl create -f examples/joint_inference/llm/llm_inference.yaml
   ```

## 检查服务状态

```
kubectl get jointinferenceservices.sedna.io
```

## 推理测试

- 通过接口传入 input_text 环境变量指定的文本，edge 侧小模型优先推理，置信度低或难例将上传 cloud 侧大模型处理。
- 推理结果可在 `/joint_inference/output` 目录下查看。

## 参考
- [Sedna 联合推理文档](https://github.com/kubeedge/sedna)
- [transformers 文档](https://huggingface.co/docs/transformers) 