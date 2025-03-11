# FA Example Readme
## 代码组织
```
├── 05_fa
│   ├── CMakeLists.txt # CMake编译文件
│   ├── README.md
│   ├── fa_kernel.cpp  # FA模板所需的模块
│   ├── gen_data.py    # 测试数据的生成脚本
│   └── main.cpp       # 主文件
```
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 第一步，首先执行`gen_data.py`，生成测试样例，测试用例需要从命令行输入。
```
python gen_data.py 1 256 8 1 128 512
# 输入参数分别对应 batch size，seqlen，qhead_num，kvhead_num，embedsize，maxseqlen
# seqlen表示输入序列长度，maxseqlen表示最大支持序列长度用于生成下三角矩阵
```
执行该命令后会在当前路径下生成data目录，包含算子的输入数据和用于精度验证的golden数据
```
├── data
│   ├── golden.bin
│   ├── k.bin
│   ├── kv_seqlen.bin
│   ├── mask.bin
│   ├── q.bin
│   ├── q_ntokens.bin
│   ├── q_seqlen.bin
│   └── v.bin
```
第二步，执行算子，这里要注意的是执行算子的输入shape和上面第一步生成数据的shape一致。
```
# cd [代码仓路径]/build/bin
./05_fa 1 256 8 1 128 512
# 此处的参数和生成数据的参数保持一致
# 完整参数为 batch seqlen qhead_num kvhead_num embedsize maxseqlen [--datapath DATA_PATH --device DEVICE_ID]，datapath默认为../../examples/05_fa/data, device默认为0。
```
执行结果如下，说明精度比对成功。
```
Running fa: batch=1, seqlen=256, qHead=8, groupNum=1, embed=128, maxSeqLen=512...
Compare success.
```