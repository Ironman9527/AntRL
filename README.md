## AntEnv（Unity 蚂蚁觅食强化学习环境）

一个基于 Unity 的多智能体蚂蚁觅食环境，内置信息素场与射线感知，提供类 Gym 的 Python 包装器（ZMQ + MessagePack）以便与各类 RL 框架集成。支持大规模并发智能体与无图形批量模式，适合强化学习、群体智能与自组织行为研究与教学演示。


### 特性

- **多智能体并发**：默认可扩展至 上千只 蚂蚁并发（RTX5090下测试可4096只）（可调 `numAgents`）。
- **信息素场建模**：提供“归巢/食物”双通道信息素网格帧（可用于可视化或作为特征）。
- **类 Gym 接口**：`reset/step/close` 与标准数据流，利于与 RL 训练脚本对接。
- **高性能通信**：Python 端为 ZMQ REP，Unity 端为 REQ，请求-应答稳态高吞吐；消息使用 MessagePack 编解码。
- **批量/无图形渲染**：支持 `-batchmode`、`-nographics`、`-perfMode` 参数，便于服务器端训练。
- **日志与 RPS 监控**：示例脚本内置每秒请求数（RPS）监控与 Unity 侧日志输出。


### 目录结构（关键部分）

```
AntRL/
├─ AntEnv/                         # 已编译的 Unity Player（Windows）
│  ├─ Ant.exe
│  └─ Ant_Data/ ...                # Unity 运行时资源
├─ GymlikeEnvWrapper/
│  ├─ ant_env.py                   # 类 Gym 环境包装器（ZMQ + MessagePack）
│  ├─ ant_types.py                 # 通信数据结构定义与序列化/反序列化
│  └─ __pycache__/ ...
├─ logs/
│  └─ ant_unity_5555.txt           # Unity 运行日志（示例）
├─ test_ant_env.py                 # 环境连通性与 RPS 测试
└─ README.md
```

### 运行环境

- 操作系统：Windows 10/11（示例使用 Win64 可执行文件）
- Python：3.8+（推荐 3.9/3.10）
- 依赖包：`numpy`、`pyzmq`、`msgpack`
- Unity：已提供编译好的 `Ant.exe`；

安装 Python 依赖（示例）：

```bash
pip install numpy pyzmq msgpack
```


### 快速开始

#### 方式 A：由 Python 启动 Unity 可执行文件

```python
from GymlikeEnvWrapper.ant_env import AntForagingEnv, create_default_config

exe_path = r'D:\Program Files (x86)\VScodeWorkSpace\GameEnvUnity\AntRL\AntEnv\Ant.exe'
config = create_default_config(num_agents=512, port=5555, executable_path=exe_path)
env = AntForagingEnv(config)

obs, info = env.reset()

# 示例：随机动作循环
import numpy as np
for _ in range(1000):
    actions = np.random.randint(0, 3, size=(env.numAgents * 2,), dtype=np.int32)
    obs, rewards, terminateds, truncateds, info = env.step(actions)

env.close()
```

#### 方式 B：先手动启动 `Ant.exe`，再用 Python 连接

命令行参数示例（Unity Player）：

```bash
Ant.exe -ip 127.0.0.1 -port 5555 -ants 1024 -maxSteps 1500 ^
  -logfile ./logs/ant_unity_5555.txt -curriculumFactor 0.1 -batchmode -nographics -perfMode
```

随后在 Python 中以 `executablePath=None` 连接（仅建立 ZMQ 端口）：

```python
from GymlikeEnvWrapper.ant_env import AntForagingEnv, create_default_config

config = create_default_config(num_agents=1024, port=5555, executable_path=None)
env = AntForagingEnv(config)
obs, info = env.reset()
```

#### 方式 C：Unity Editor 运行

- 在 Unity Editor 打开工程，运行场景（确保使用相同的 `-ip/-port` 设置或在脚本中对齐）。
- Python 端以 `executablePath=None` 连接（与方式 B 相同）。


### Python API 概览

- `AntForagingEnv(config: Dict[str, Any])`：环境实例，提供 `reset/step/close/restart`。
- `create_default_config(num_agents=4, port=5555, executable_path=None, log_dir="ant_logs")`：创建默认配置。
- 关键配置项：
  - **ip/port**：ZMQ 绑定地址（Python 端为服务器 `REP`）。
  - **numAgents**：智能体数量（影响动作数组长度与 Unity 端实例化数）。
  - **numFoods**: 食物堆的数量。
  - **maxSteps**：Unity 内部最大步数（由 Player 接收并生效）。
  - **timeout**：通信超时（秒）。
  - **executablePath**：Unity Player 路径（为 `None` 时不启动外部进程）。
  - **bg**:后台运行参数，加上后会请求会无阻塞。
  - **batchmode/nographics**：无界面运行。
  - **logfile**：Unity 侧日志文件路径（通过命令行传入）。
  - **curriculum_factor**：课程因子，控制生成的食物与蚁巢之间的距离。


### 动作空间

- 动作张量形状：`(numAgents*2,)`，按智能体拼接，每个智能体 2 维：
  - 第 1 维——移动：离散值 `{0,1,2}`，在环境内部映射为 `{-1.0, 0.0, 1.0}`（后退/停止/前进）。
  - 第 2 维——转向：离散值 `{0..14}`，映射为下列 15 个连续值：

```text
index:  0    1    2    3    4    5     6     7     8     9    10    11    12    13    14
value: -1.0 -0.8 -0.6 -0.4 -0.2 -0.1 -0.05  0.0  0.05  0.1  0.2  0.4  0.6  0.8  1.0
```

- Python 内部会将 `(move, turn)` 成对重排并转换为 `{agentId: [moveInput, rotateInput]}` 字典发送给 Unity。


### 观测与元数据

- `AntObservation`（每只蚂蚁一条，关键字段）：
  - `agentId`、`position`、`rotation`、`velocity`、`angularVelocity`
  - `isCarryingFood`、`pickedFood`、`deliveredFood`、`outOfBounds`、`isColliding`
  - `distanceFromHome`、`directionToHome`
  - `flatRaycastFeatures`（射线特征，已在 Unity 侧预处理为扁平向量）
  - `gridIndexX/gridIndexY`（信息素网格索引）
  - `done`（个体完成标记，如使用）

- `PheromoneGridFrame`（可选帧级别信息素网格）：
  - `width/height`（默认 512×512）、`cellSize`、`originWorld`
  - `toHome`、`toFood`（二维 `float32` 数组，形状 `(height, width)`）

- 环境级 `info`：
  - `step`、`envDone`、`foodRemaining`、`foodTotal`、`episodeId`
  - `current_episode`、`current_step`、`curriculum_factor`


### 通信协议（Python 为 ZMQ REP，Unity 为 ZMQ REQ）

- 编解码：MessagePack（二进制，高效紧凑）。
- 数据流：
  1. `reset()` 时，Python 首先从 Unity 接收一帧初始 `AntBatchDataToPython`；
  2. `step(actions)` 时，Python 发送 `AntBatchActionFromPython`（包含所有蚂蚁动作与 `curriculumFactor`）；
  3. Unity 处理后返回新一帧 `AntBatchDataToPython`；
  4. 如 `envDone=True`，表示该 Episode 结束（可选择 `reset()` 或继续 `step()` 取决于上层逻辑）。

- 超时与健壮性：
  - Python 侧对 `send/recv` 均设置了超时与高水位（HWM）保护，避免阻塞与堆积。
  - 发生超时会抛出 `TimeoutError`，可在上层重试或调用 `restart()` 重建通信。


### Unity Player 命令行参数

- `-ip <string>`：ZMQ 地址（与 Python 端保持一致）。
- `-port <int>`：ZMQ 端口（默认 `5555`）。
- `-ants <int>`：智能体数量（与 Python `numAgents` 对齐）。
- `-maxSteps <int>`：Unity 内部最大步数（用于终止条件等）。
- `-logfile <path>`：Unity 日志输出路径（示例 `./logs/ant_unity_5555.txt`）。
- `-curriculumFactor <float>`：课程学习因子（Python 端也会发送同名字段）。
- `-bg`：后台标记（如需）。
- `-batchmode -nographics -perfMode`：批量/无图形/性能优先模式。


### 示例：连通性 + RPS 监控

可直接运行仓库内的 `test_ant_env.py`：

```bash
python test_ant_env.py
```

其内部会：
- 启动/连接 Unity Player；
- 循环以随机动作 `step`；
- 每 5 秒打印窗口内平均 RPS；
- Ctrl+C 退出时自动 `close()`。


### 性能与规模

- 单机多智能体：默认配置测试通过 512–1024 体规模（取决于硬件与图形模式）。
- 建议在训练中打开 `-batchmode -nographics -perfMode` 并合理设置 `timeout/HWM`。


### 日志与排错

- Unity 侧日志文件默认位于 `./logs/ant_unity_<port>.txt`（可通过命令行覆盖）。
- 若出现通信超时：
  - 检查 `ip/port` 是否一致、端口是否被占用；
  - 降低 `numAgents` 或增大 `timeout`；
  - 调用 `env.restart()` 重建通信。

### 相机控制

- 通过鼠标控制主相机旋转
- 通过WASD控制主相机移动
- 按空格会附身在最近的蚂蚁身上，再次按下脱离蚂蚁

### 其他

- 地图设置为200*200的一个方形区域，蚂蚁越界时会在蚁巢重生 

### 开源许可

请为仓库选择开源许可证（推荐 MIT/Apache-2.0）。如未提供，将默认视为“保留所有权利”。


### 引用与致谢

- 若本项目对您的研究或产品有帮助，欢迎在论文与 README 中引用本仓库。
- 欢迎提 Issue/PR 共同完善
