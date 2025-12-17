### vllm_ascend + xlite 在线服务性能测试及对比

1. **快速开始**
```
# 安装vllm_ascend, 可参考https://github.com/vllm-project/vllm-ascend/blob/main/README.md

# 安装xlite
pip install xlite
```

2. **启动在线服务**
```
# xlite可选择decode only或full其中一种模式启动在线服务：
# a. 配置xlite decode only模式，设置--additional-config='{"xlite_graph_config": {"enabled": true}}'
cd ./tests/e2e/
bash online_server_xlite_decode_only.sh 8080 server.log

# b. 配置xlite full模式，设置--additional-config='{"xlite_graph_config": {"enabled": true, "full_mode": true}}'
cd ./tests/e2e/
bash online_server_xlite_full.sh 8080 server.log
```

3. **启动在线压测**
```
# 运行压测脚本
cd ./tests/e2e/
bash online_server_test.sh
```
压测脚本中，输入序列和输出序列的默认 token 数均为 512，在用户并发数 1、16、32、48、64、100、150 及 200 的条件下，分别测试并采集性能数据。

4. **性能对比**
```
# 用户可以直接运行性能对比脚本，进行aclgraph、xlite_full和xlite_decode_only三种模式下在线服务的性能对比
cd ./tests/e2e/
bash online_server_compare.sh

# 解析性能对比数据
# 入参为aclgraph、xlite_full和xlite_decode_only的压测结果保存路径，对比结果保存在./benchmark_comparison.log
cd ./tests/e2e/
python process_data.py ./result_input_512_output_512_aclgraph ./result_input_512_output_512_xlite_full ./result_input_512_output_512_xlite_decode_only
```