"""
蚂蚁觅食环境的数据类型定义和序列化工具

定义了与Unity端通信的所有数据结构，基于MessagePack序列化协议。
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

import msgpack
import numpy as np

@dataclass
class RaycastInfo:
    """射线检测信息"""
    tag: str = ""  # 碰撞物体的标签
    point: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # 碰撞点
    normal: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # 法线
    distance: float = 0.0  # 距离

@dataclass
class AntObservation:
    """蚂蚁观察数据"""
    # 基本信息
    agentId: int = 0
    
    # 位置和运动信息
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angularVelocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    
    # 蚂蚁状态
    isCarryingFood: bool = False
    distanceFromHome: float = 0.0
    directionToHome: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    
    # 射线检测数据
    flatRaycastFeatures: List[float] = field(default_factory=list)
    
    # 当前输入
    currentInputs: List[float] = field(default_factory=lambda: [0.0, 0.0])
    
    # 事件标志
    pickedFood: bool = False
    pickedPileRemainRatio: float = 0.0
    deliveredFood: bool = False
    outOfBounds: bool = False
    isColliding: bool = False  # 是否正在碰撞
    
    # 信息素网格索引
    gridIndexX: int = 0
    gridIndexY: int = 0
    
    # 完成状态
    done: bool = False

@dataclass
class PheromoneGridFrame:
    """信息素网格帧数据"""
    version: int = 0
    step: int = 0
    width: int = 256
    height: int = 256
    cellSize: float = 0.39
    originWorld: List[float] = field(default_factory=lambda: [-50.0, 0.0, -50.0])
    toHome: np.ndarray = field(default_factory=lambda: np.zeros((256, 256), dtype=np.float32))
    toFood: np.ndarray = field(default_factory=lambda: np.zeros((256, 256), dtype=np.float32))

@dataclass
class AntBatchDataToPython:
    """发送给Python的批量数据"""
    observations: List[AntObservation] = field(default_factory=list)
    step: int = 0
    envDone: bool = False
    foodRemaining: int = 0
    foodTotal: int = 0
    episodeId: int = 0
    gridFrame: Optional[PheromoneGridFrame] = None

@dataclass
class AntAction:
    """蚂蚁动作"""
    agentId: int = 0
    moveInput: float = 0.0  # [-1, 1] 前进/后退
    rotateInput: float = 0.0  # [-1, 1] 左转/右转

@dataclass
class AntBatchActionFromPython:
    """从Python发送的批量动作"""
    actions: List[AntAction] = field(default_factory=list)
    curriculumFactor: float = 0.1

def serialize_data(data: Any) -> bytes:
    """序列化数据为MessagePack格式
    
    Args:
        data: 要序列化的数据（通常是AntBatchActionFromPython）
        
    Returns:
        序列化后的字节数据
        
    Raises:
        Exception: 序列化失败时抛出
    """
    try:
        if isinstance(data, AntBatchActionFromPython):
            # 转换为字典格式，匹配Unity端的MessagePack结构
            actions_dict = [
                {
                    "agentId": action.agentId,
                    "moveInput": action.moveInput,
                    "rotateInput": action.rotateInput
                }
                for action in data.actions
            ]
            
            data_dict = {
                "actions": actions_dict,
                "curriculumFactor": data.curriculumFactor
            }
            return msgpack.packb(data_dict)
        return msgpack.packb(data)
    except Exception as e:
        raise Exception(f"序列化失败: {e}") from e

def deserialize_data(data: bytes) -> AntBatchDataToPython:
    """反序列化MessagePack数据
    
    Args:
        data: 序列化的字节数据
        
    Returns:
        AntBatchDataToPython对象
        
    Raises:
        Exception: 反序列化失败时抛出
    """
    try:
        raw_data = msgpack.unpackb(data, raw=False)
        
        # 解析观察数据
        observations = _parse_observations(raw_data.get("observations", []))
        
        # 解析信息素网格帧
        grid_frame = _parse_grid_frame(raw_data.get("gridFrame"))
        
        return AntBatchDataToPython(
            observations=observations,
            step=raw_data.get("step", 0),
            envDone=raw_data.get("envDone", False),
            foodRemaining=raw_data.get("foodRemaining", 0),
            foodTotal=raw_data.get("foodTotal", 0),
            episodeId=raw_data.get("episodeId", 0),
            gridFrame=grid_frame
        )
        
    except Exception as e:
        raise Exception(f"反序列化失败: {e}") from e


def _parse_observations(obs_list: List[Dict]) -> List[AntObservation]:
    """解析观察数据列表"""
    observations = []
    for obs_data in obs_list:
        # 解析射线检测数据
        raycast_data = [
            RaycastInfo(
                tag=ray_data.get("tag", ""),
                point=ray_data.get("point", [0.0, 0.0, 0.0]),
                normal=ray_data.get("normal", [0.0, 0.0, 0.0]),
                distance=ray_data.get("distance", 0.0)
            )
            for ray_data in obs_data.get("raycastData", [])
        ]
        
        observation = AntObservation(
            agentId=obs_data.get("agentId", 0),
            position=obs_data.get("position", [0.0, 0.0, 0.0]),
            rotation=obs_data.get("rotation", [0.0, 0.0, 0.0]),
            velocity=obs_data.get("velocity", [0.0, 0.0, 0.0]),
            angularVelocity=obs_data.get("angularVelocity", [0.0, 0.0, 0.0]),
            isCarryingFood=obs_data.get("isCarryingFood", False),
            distanceFromHome=obs_data.get("distanceFromHome", 0.0),
            directionToHome=obs_data.get("directionToHome", [0.0, 0.0, 0.0]),
            flatRaycastFeatures=obs_data.get("flatRaycastFeatures", []),
            currentInputs=obs_data.get("currentInputs", [0.0, 0.0]),
            pickedFood=obs_data.get("pickedFood", False),
            pickedPileRemainRatio=obs_data.get("pickedPileRemainRatio", 0.0),
            deliveredFood=obs_data.get("deliveredFood", False),
            outOfBounds=obs_data.get("outOfBounds", False),
            isColliding=obs_data.get("isColliding", False),
            gridIndexX=obs_data.get("gridIndexX", 0),
            gridIndexY=obs_data.get("gridIndexY", 0),
            done=obs_data.get("done", False)
        )
        observations.append(observation)
    return observations


def _parse_grid_frame(gf_data: Optional[Dict]) -> Optional[PheromoneGridFrame]:
    """解析信息素网格帧"""
    if gf_data is None:
        return None
    
    width = gf_data.get("width", 256)
    height = gf_data.get("height", 256)
    
    return PheromoneGridFrame(
        version=gf_data.get("version", 0),
        step=gf_data.get("step", 0),
        width=width,
        height=height,
        cellSize=gf_data.get("cellSize", 0.39),
        originWorld=gf_data.get("originWorld", [-50.0, 0.0, -50.0]),
        toHome=np.array(gf_data.get("toHome", []), dtype=np.float32).reshape((height, width)),
        toFood=np.array(gf_data.get("toFood", []), dtype=np.float32).reshape((height, width))
    )

def create_actions_response(actions: Dict[int, List[float]], curriculum_factor: float = 0.1) -> AntBatchActionFromPython:
    """创建动作响应数据
    
    Args:
        actions: 智能体ID到动作的映射 {agent_id: [move_input, rotate_input]}
        curriculum_factor: 课程学习因子
        
    Returns:
        AntBatchActionFromPython对象
    """
    action_list = [
        AntAction(
            agentId=agent_id,
            moveInput=float(action[0]),
            rotateInput=float(action[1])
        )
        for agent_id, action in actions.items()
    ]
    
    return AntBatchActionFromPython(
        actions=action_list,
        curriculumFactor=curriculum_factor
    )


# 数据提取工具函数
def extract_ant_data(batch_data: AntBatchDataToPython) -> List[AntObservation]:
    """提取蚂蚁观察数据
    
    Args:
        batch_data: 批量数据
        
    Returns:
        观察数据列表
    """
    return batch_data.observations if batch_data.observations else []


def get_ant_position(ant_obs: AntObservation) -> np.ndarray:
    """获取蚂蚁位置
    
    Args:
        ant_obs: 蚂蚁观察数据
        
    Returns:
        位置向量 [x, y, z]
    """
    return np.array(ant_obs.position, dtype=np.float32)


def get_ant_home_direction(ant_obs: AntObservation) -> np.ndarray:
    """获取蚂蚁回家方向
    
    Args:
        ant_obs: 蚂蚁观察数据
        
    Returns:
        方向向量 [x, y, z]
    """
    return np.array(ant_obs.directionToHome, dtype=np.float32)


def calculate_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """计算两点之间的距离
    
    Args:
        pos1: 位置1
        pos2: 位置2
        
    Returns:
        欧几里得距离
    """
    return float(np.linalg.norm(pos1 - pos2))


def extract_pheromone_data(batch_data: AntBatchDataToPython) -> Optional[PheromoneGridFrame]:
    """提取信息素网格数据
    
    Args:
        batch_data: 批量数据
        
    Returns:
        信息素网格帧（如果有）
    """
    return batch_data.gridFrame


def extract_environment_metadata(batch_data: AntBatchDataToPython) -> Dict[str, Any]:
    """提取环境元数据
    
    Args:
        batch_data: 批量数据
        
    Returns:
        元数据字典
    """
    return {
        "step": batch_data.step,
        "envDone": batch_data.envDone,
        "foodRemaining": batch_data.foodRemaining,
        "foodTotal": batch_data.foodTotal,
        "episodeId": batch_data.episodeId
    }
