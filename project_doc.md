# pi0 项目说明文档

## Installation

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.

## 数据集转换脚本
```
uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /home/zme/data/robot_data/fold_clothes/ --repo-id zme/fold_clothes
```
转换后lerobot数据集会保存在： /home/zme/.cache/huggingface/lerobot/zme/fold_clothes

### 计算数据集的统计信息
```
uv run scripts/compute_norm_stats.py --config-name pi0_zme --repo_id zme/fold_clothes
```
结果会保存在assets/pi0_zme/zme/fold_clothes/norm_stats.json
将这个文件复制到 /home/zme/.cache/huggingface/lerobot/zme/fold_clothes/norm_stats.json

## 训练
```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_zme --exp-name=experiment_fold_clothes --overwrite --batch-size 32
```

## 推理
```
cd openpi
export PYTHON_PATH=/opt/ros/humble/lib/python3.10/site-packages         # 根据电脑实际情况修改ros2的python包路径
```

1. 先在client窗口运行simple_client,这个操作是为了重置uv 环境。
```
http_proxy= https_proxy= PYTHON_PATH=/opt/ros/humble/lib/python3.10/site-packages uv run --python 3.10 examples/simple_client/main.py --env ZME
```
2. 在一个新的teiminal 运行server
```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_zme --policy.dir=/home/zme/model/experiment_fold_clothes_bs32/49999
```

3. 在client窗口运行正式client
```
ROS_DOMAIN_ID=10 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp http_proxy= https_proxy= PYTHON_PATH=/opt/ros/humble/lib/python3.10/site-packages uv run --python 3.10 examples/zme_client/main.py --env ZME
```
