# BODex 编译标准修复

文件：`/home/ubuntu/DATA2/workspace/xmh/BODex/src/curobo/geom/cpp/setup.py`

修改：
- `-std=c++11` -> `-std=c++14`

原因：在当前 `objdex` + `coal/boost` 组合下，`coal_openmp_wrapper` 需要 C++14 才能稳定编译通过。
