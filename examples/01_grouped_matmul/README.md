# GroupedMatmul Example Readme
## 代码组织
```
├── 02_grouped_matmul
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── grouped_matmul.cpp # 主文件
```
## 使用示例
因为GroupedMatmul参数较多，所以该示例直接在代码中承载输出参数列表`groupList`。
相关输入配置代码如下，具体详见[grouped_matmul.cpp](grouped_matmul.cpp)。
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# cd [代码仓路径]/build/bin
./02_grouped_matmul
```
执行结果如下，说明精度比对成功。
```
Compare success.
```