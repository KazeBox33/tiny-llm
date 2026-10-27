# Week 2 Day 2 Task 1：四位编码与按需恢复嵌入行

日期：2026-10-27。

对应上游教材 `da3e841` 的 Represent Weights With Fewer Bits、Group-Wise Affine Quantization、Packed Storage Layout，以及 Task 1 的嵌入行查找部分。
本节没有完成整个 Day 2，也没有迁移新版课程的检查点选择器或完整验收入口。

## 本次实现范围

现有 `QuantizedEmbedding.__call__` 已经先选择压缩权重行，但恢复数值时直接调用 `mx.dequantize`。
新版教材明确要求在 `dequantize_weights` 中用基础 MLX 数组运算显式解包，再供嵌入行查找调用。

因此本次新增 `src/tiny_llm/quantize.py::dequantize_weights`，并让 `src/tiny_llm/embedding.py::QuantizedEmbedding.__call__` 调用它。
保留原有 `QuantizedWeights` 加载路径、量化投影包装和原生量化矩阵乘实现。
没有修改 C++/Metal 扩展，因此不需要重建扩展。
本笔记随这一步的源码和对应测试一同版本管理，提交与推送状态以 Git 记录为准。

## 这一部分要解决的问题

CS336 中可以直接用词元编号索引浮点嵌入表。
四位模型保存的是压缩整数和量化信息，需要先恢复数值，才能把向量交给后面的网络。
输入嵌入阶段只需要当前词元对应的行，保留其余词表行的压缩表示可以避免展开整张大表。

Qwen3-4B 的词表有 151936 行，每行 2560 个元素。
展开整表为 BF16 需要 777,912,320 字节，约 777.9 MB。
单个词元最终只产生 2560 个 BF16 元素，即 5120 字节；此外仍有压缩行选择和 FP32 中间数组的开销。
这两个数字描述逻辑张量大小，不是本节测出的活跃内存峰值。

## 四位编码怎样表示浮点权重

本课程使用 W4A16：权重用四位编码保存，激活和输出为 BF16。
四位编码是 0 到 15 的无符号整数，恢复公式为：

```text
恢复后的权重 = 编码 × 存储的缩放值 + 存储的偏移值
w_hat = q × scale + bias
```

这里的偏移值是每个量化分组的恢复参数。
每行每 128 个权重共享一组 BF16 缩放和偏移信息。
缩放值可以为负，必须保留其符号，不能根据假设的最小值和最大值顺序重新构造参数。

例如 `scale = 0.125, bias = -0.5` 时，编码 0、4、15 分别恢复为 -0.5、0、1.375。
反过来使用 `scale = -0.125, bias = 1.375` 时，编码 0 对应 1.375，编码 15 对应 -0.5。

一个 uint32 从低位到高位可以保存八个四位编码：

```text
编码顺序：          [0, 1, 2, 3, 4, 5, 6, 7]
压缩为十六进制整数：0x76543210
第 j 个编码：       (packed >> (4 × j)) & 0xF
```

右移把目标编码移到最低位，掩码 0xF 保留最低四位。
编码顺序是从低位向高位读取。

## 先选行，再恢复

本地模型原始嵌入权重的形状为：

```text
压缩编码 weight：[151936, 320]，uint32；2560 / 8 = 320
缩放值 scales： [151936, 20]， BF16；2560 / 128 = 20
偏移值 biases： [151936, 20]， BF16
```

输入词元编号 `[[5, 1, 5]]` 时，先分别索引三种数组：

```text
词元编号：                [1, 3]
选择的压缩编码：          [1, 3, 320]
选择的缩放和偏移：        [1, 3, 20]
每个整数取出八个编码后：  [1, 3, 320, 8]
按量化分组重排：          [1, 3, 20, 128]
恢复并合并最后两维：      [1, 3, 2560]
```

重复词元仍保留各自的位置，本节没有额外引入去重或嵌入结果缓存。
数组操作由 MLX 建图并在求值时执行，模型路径没有转成 NumPy 或在 CPU 上逐个解包。

`dequantize_weights` 用广播将每组一个缩放和偏移应用于组内 128 个编码，无须显式复制参数 128 次。
恢复时将编码、缩放、偏移转成 FP32，计算结束统一转回 BF16。
输入必须符合当前 W4/G128、uint32 编码及 BF16 量化信息的约定；不匹配时直接报错。

这里显式恢复的是输入嵌入选中的行。
输出阶段的共享词表投影仍通过 `QuantizedEmbedding.as_linear` 使用完整压缩矩阵计算所有候选词元分数，两种用途不能混淆。

## 精度选择依据

先用三个固定随机种子 0、2、7，各生成 7 行、256 列的 BF16 权重，并由 MLX 按 W4/G128 量化。
将两种基础数组表达式与本地 MLX 0.29.1 的 `mx.dequantize` 比较：

| 恢复计算方式 | 三个种子的最大绝对误差 |
|---|---|
| FP32 乘加，最后转回 BF16 | 均为 0 |
| BF16 分步乘加 | 均为 0.015625 |

因此本次使用 FP32 中间计算，避免乘法结果先舍入为 BF16。
这与本机 BF16 恢复路径的数值结果一致；不是任意改变精度以放宽误差容忍度。

## 测试与实际模型检查

新增 `tests/test_week_2_quantized_embedding.py`，执行：

```bash
pdm run ruff check src/tiny_llm/quantize.py src/tiny_llm/embedding.py \
  tests/test_week_2_quantized_embedding.py
HF_HUB_OFFLINE=1 pdm run python -m pytest tests/test_week_2_quantized_embedding.py -q
HF_HUB_OFFLINE=1 pdm run python -m pytest tests/test_week_2_model.py -q
```

结果：静态检查通过，新测试 7 项通过，已有缓存模型测试 11 项通过。

新测试覆盖：

- 手工四位编码及独立恢复期望值，核对编码顺序、正负缩放、分组边界和输出精度。
- 三个随机种子的结果与 MLX 内置恢复结果逐元素完全相等。
- 带重复词元的预填充、uint32 单词元解码、二维批输入；核对恢复函数只接收到选中行。
- 测试期间禁止模型路径调用 `mx.dequantize`，确保实际执行显式解包实现。

另外离线读取已有的 Qwen/Qwen3-4B-MLX-4bit，快照为 `52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495`。
将新嵌入实现与 MLX 对相同选中行的恢复进行比较：

| 词元编号与类型 | 输出形状 | 最大绝对误差 | 选中量化组中的负缩放数量 |
|---|---|---:|---:|
| `[[0, 1, 42, 151935]]`，int32 | `[1, 4, 2560]` | 0 | 43 |
| `[[5, 1, 5]]`，uint32 | `[1, 3, 2560]` | 0 | 36 |

形状和 BF16 类型检查通过，选中行与 MLX 逐元素完全相等。
本节没有进行延迟或峰值内存对比，不宣称手写恢复比内置算子更快。

## 下一部分

继续 Week 2 Day 2 Task 1 的输出投影路径：解释 `QuantizedEmbedding.as_linear`、`quantized_linear` 如何使用同一份压缩权重，再进入 Task 2 的原生算子输入检查和延迟执行边界。
已有代码保留，按新版教材的约定核对后，再选择需要补齐的小步骤。

## 参考

- 对齐的教材：`work/upstream-tiny-llm/book/src/week2-02-quantize-model.md`，提交 `da3e841`。
- 已安装 MLX 0.29.1 的 `mlx/nn/layers/quantized.py::QuantizedEmbedding`，用于核对先选行的数据流。
- [MLX 解量化接口](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.dequantize.html)。
- [MLX 0.29.1 Metal 量化内核](https://github.com/ml-explore/mlx/blob/v0.29.1/mlx/backend/metal/kernels/quantized.h)。
