"""
蚂蚁觅食环境 - Gym包装器

提供Unity蚂蚁觅食环境的标准Gym接口：
- ZMQ通信（REP模式）
- 可选Unity可执行文件启动
- 标准reset/step/close接口
"""
import os
import time
import logging
import atexit
import subprocess
from typing import Optional, Dict, Any, List

import zmq
import numpy as np

from GymlikeEnvWrapper.ant_types import (
    serialize_data, deserialize_data, create_actions_response,
    AntBatchDataToPython, extract_ant_data,
    extract_environment_metadata
)

logger = logging.getLogger(__name__)


class AntForagingEnv():
    """蚂蚁觅食Gym环境"""

    def __init__(self, config: Dict[str, Any]):
        """初始化环境
        
        Args:
            config: 环境配置字典
        """
        # 基础配置
        self.config = config
        self.ip = config.get("ip", "127.0.0.1")
        self.port = config.get("port", 5555)
        self.numAgents = config.get("numAgents", 1024)
        self.numFoods = config.get("numFoods", 1)
        self.maxSteps = config.get("maxSteps", 5000)
        self.timeout = config.get("timeout", 30)
        self.executablePath = config.get("executablePath", None)
        self.log_dir = config.get("log_dir", "ant_logs")
        self.bg = config.get("bg", False)
        self.batchmode = config.get("batchmode", False)
        self.nographics = config.get("nographics", False)
        self.logfile = config.get("logfile", f"./logs/ant_unity_{self.port}.txt")

        # 运行时状态
        self.latest_batch_data: Optional[AntBatchDataToPython] = None
        self.previous_batch_data: Optional[AntBatchDataToPython] = None
        self.curriculum_factor: float = 0.1
        self.current_episode = 0
        self.current_step = 0
        self.unity_process = None

        # 初始化通信和Unity
        self._setup_communication()
        self._launch_unity_if_needed()

    def _setup_communication(self) -> None:
        """设置ZMQ通信"""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        
        # 设置超时（毫秒）
        timeout_ms = self.timeout * 1000 * 2
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        logger.info(f"Set RCV/SND timeout: {self.timeout}s")
        
        # 设置其他选项
        try:
            self.socket.setsockopt(zmq.LINGER, 0)  # 关闭时立即丢弃未发送消息
            self.socket.setsockopt(zmq.SNDHWM, 2)  # 发送队列最多2条
            self.socket.setsockopt(zmq.RCVHWM, 2)  # 接收队列最多2条
            logger.debug(f"Set LINGER=0, SNDHWM=2, RCVHWM=2")
        except Exception as e:
            logger.warning(f"Failed to set some socket options: {e}")
        
        self.socket.bind(f"tcp://{self.ip}:{self.port}")
        logger.info(f"Ant server bound to {self.ip}:{self.port}")
        
        # 验证超时设置
        actual_rcv = self.socket.getsockopt(zmq.RCVTIMEO)
        actual_snd = self.socket.getsockopt(zmq.SNDTIMEO)
        logger.info(f"Actual timeouts - RCV: {actual_rcv}ms, SND: {actual_snd}ms")

    def _launch_unity_if_needed(self) -> None:
        """如果提供了可执行文件路径，启动Unity"""
        if not self.executablePath:
            logger.info("Unity editor mode (no external process)")
            return
        
        exe = os.path.abspath(self.executablePath)
        if not os.path.exists(exe):
            raise FileNotFoundError(f"Unity executable not found: {exe}")
        
        # 构建启动命令
        cmd = [
            exe, "-ip", self.ip, "-port", str(self.port),
            "-ants", str(self.numAgents), "-foods", str(self.numFoods), "-maxSteps", str(self.maxSteps),
            "-logfile", str(self.logfile), "-curriculumFactor", str(self.curriculum_factor)
        ]
        if self.bg:
            cmd.append("-bg")
        if self.batchmode:
            cmd.append("-batchmode")
        if self.nographics:
            cmd.extend(["-nographics", "-perfMode"])
        
        logger.info("Launching Unity: %s", " ".join(cmd))
        try:
            self.unity_process = subprocess.Popen(cmd)
            atexit.register(self.close)
            time.sleep(3)  # 等待Unity启动
        except Exception as e:
            raise Exception(f"Failed to launch Unity: {e}") from e

    def reset(self,):
        self.current_step = 0
        self.current_episode += 1
        try:
            t0 = time.monotonic()
            msg = self.socket.recv()
            t1 = time.monotonic()
            dt = t1 - t0
            if dt > 2.0:
                logger.warning(f"ZMQ reset recv slow: {dt:.3f}s (port {self.port})")
        except zmq.error.Again:
            logger.error(f"ZMQ reset recv timeout after {self.timeout}s (port {self.port})")
            raise TimeoutError(f"Waiting for Unity initial state timed out ({self.timeout}s)")
        self.latest_batch_data = deserialize_data(msg)
        self.previous_batch_data = self.latest_batch_data

        observations = extract_ant_data(self.latest_batch_data)
        info = self._get_info()
        return observations, info

    def step(self, actions: np.ndarray):
        """执行一步环境交互
        
        Args:
            actions: 动作数组，形状为(num_agents*2,)，值为0/1/2
            
        Returns:
            (observations, rewards, terminateds, truncateds, infos)
        """
        self.current_step += 1
        self.previous_batch_data = self.latest_batch_data

        # 转换动作：MultiDiscrete(0,1,2) -> (-1,0,1)
        actions_dict = self._convert_actions(actions)

        # 发送动作并接收新状态
        response = create_actions_response(actions_dict, self.curriculum_factor)
        
        # 发送数据（带超时监控）
        try:
            serialized = serialize_data(response)
            logger.debug(f"[Step {self.current_step}] Sending {len(serialized)} bytes, {len(actions_dict)} actions")
            
            t0 = time.monotonic()
            self.socket.send(serialized)
            send_time = time.monotonic() - t0
            
            if send_time > 1.0:
                logger.warning(f"[Step {self.current_step}] Send slow: {send_time:.3f}s (port {self.port})")
            logger.debug(f"[Step {self.current_step}] Send OK in {send_time:.3f}s")
            
        except zmq.error.Again:
            logger.error(f"[Step {self.current_step}] SEND TIMEOUT after {self.timeout}s (port {self.port})")
            raise TimeoutError(f"Send to Unity timed out ({self.timeout}s)")
        except Exception as e:
            logger.error(f"[Step {self.current_step}] Send error: {type(e).__name__}: {e}")
            raise
        
        # 接收响应（带超时监控）
        try:
            t0 = time.monotonic()
            msg = self.socket.recv()
            recv_time = time.monotonic() - t0
            
            if recv_time > 2.0:
                logger.warning(f"[Step {self.current_step}] Recv slow: {recv_time:.3f}s (port {self.port})")
            logger.debug(f"[Step {self.current_step}] Recv OK in {recv_time:.3f}s, {len(msg)} bytes")
            
        except zmq.error.Again:
            logger.error(f"[Step {self.current_step}] RECV TIMEOUT after {self.timeout}s (port {self.port})")
            raise TimeoutError(f"Recv from Unity timed out ({self.timeout}s)")
        except Exception as e:
            logger.error(f"[Step {self.current_step}] Recv error: {type(e).__name__}: {e}")
            raise
        
        self.latest_batch_data = deserialize_data(msg)

        observations = extract_ant_data(self.latest_batch_data)

        # 简化奖励和终止标志（实际奖励在外部计算）
        n = len(observations)
        rewards = np.zeros(n, dtype=np.float32)
        terminateds = np.zeros(n, dtype=bool)
        truncateds = terminateds
        infos = self._get_info()

        return observations, rewards, terminateds, truncateds, infos
    
    def _convert_actions(self, actions: np.ndarray) -> Dict[int, List[float]]:
        """转换动作格式
        
        移动动作 (0,1,2) -> (-1, 0, 1)
        转向动作 (0~14) -> 15个离散值：
            0->-1.0, 1->-0.8, 2->-0.6, 3->-0.4, 4->-0.2, 5->-0.1, 6->-0.05,
            7->0.0,
            8->0.05, 9->0.1, 10->0.2, 11->0.4, 12->0.6, 13->0.8, 14->1.0
        """
        actions_dict: Dict[int, List[float]] = {}
        
        # 获取智能体ID
        prev_obs_all = extract_ant_data(self.latest_batch_data) if self.latest_batch_data else []
        agent_ids = [getattr(o, 'agentId', i) for i, o in enumerate(prev_obs_all[:self.numAgents])]
        max_actions = min(self.numAgents, len(actions) // 2, len(agent_ids))
        
        if max_actions > 0:
            # 转向动作映射表 (15维)
            turn_mapping = np.array([
                -1.0, -0.8, -0.6, -0.4, -0.2, -0.1, -0.05,  # 0~6
                0.0,                                          # 7
                0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0           # 8~14
            ], dtype=np.float32)
            
            # 分离移动和转向动作
            actions_reshaped = actions[:max_actions*2].reshape(max_actions, 2)
            move_actions = actions_reshaped[:, 0].astype(np.int32)
            turn_actions = actions_reshaped[:, 1].astype(np.int32)
            
            # 移动: (0,1,2) -> (-1,0,1)
            move_values = (move_actions.astype(np.float32) - 1.0)
            
            # 转向: (0~14) -> 查表映射
            turn_actions_clipped = np.clip(turn_actions, 0, 14)  # 安全边界检查
            turn_values = turn_mapping[turn_actions_clipped]
            
            # 合并为 [移动, 转向]
            actions_converted = np.stack([move_values, turn_values], axis=1)
            
            # 转换为字典
            agent_ids_array = np.array(agent_ids[:max_actions], dtype=np.int32)
            actions_dict = dict(zip(agent_ids_array.tolist(), actions_converted.tolist()))
        
        return actions_dict

    def _get_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        if self.latest_batch_data:
            info.update(extract_environment_metadata(self.latest_batch_data))
        info["current_episode"] = self.current_episode
        info["current_step"] = self.current_step
        info["curriculum_factor"] = self.curriculum_factor
        return info

    def close(self) -> None:
        """关闭环境和资源"""
        logger.info("Closing AntForagingEnv...")
        
        # 关闭ZMQ socket
        if hasattr(self, 'socket') and self.socket:
            self.socket.close()
            self.socket = None
        if hasattr(self, 'context') and self.context:
            self.context.term()
            self.context = None
        
        # 关闭Unity进程
        if self.unity_process and self.unity_process.poll() is None:
            try:
                self.unity_process.terminate()
                self.unity_process.wait(timeout=10)
                if self.unity_process.poll() is None:
                    self.unity_process.kill()
                    self.unity_process.wait(timeout=5)
            except Exception:
                pass
        self.unity_process = None
        
        logger.info("AntForagingEnv closed")

    def restart(self) -> None:
        """重启环境（关闭当前Unity进程并启动新的）"""
        logger.info(f"Restarting AntForagingEnv on port {self.port}...")
        
        # 先关闭现有资源
        # 关闭ZMQ socket
        if hasattr(self, 'socket') and self.socket:
            self.socket.close()
            self.socket = None
        if hasattr(self, 'context') and self.context:
            self.context.term()
            self.context = None
        
        # 关闭Unity进程
        if self.unity_process and self.unity_process.poll() is None:
            try:
                logger.info(f"Terminating Unity process (PID: {self.unity_process.pid})...")
                self.unity_process.terminate()
                self.unity_process.wait(timeout=10)
                if self.unity_process.poll() is None:
                    logger.warning("Force killing Unity process...")
                    self.unity_process.kill()
                    self.unity_process.wait(timeout=5)
                logger.info("Unity process terminated successfully")
            except Exception as e:
                logger.error(f"Error terminating Unity process: {e}")
        self.unity_process = None
        
        # 等待一小段时间确保端口释放
        time.sleep(2)
        
        # 重新初始化通信和Unity
        logger.info("Re-initializing communication and Unity...")
        self._setup_communication()
        self._launch_unity_if_needed()
        
        # 重置运行时状态
        self.latest_batch_data = None
        self.previous_batch_data = None
        self.current_step = 0
        
        logger.info("AntForagingEnv restarted successfully")

def create_default_config(num_agents: int = 4,
                          port: int = 5555,
                          executable_path: Optional[str] = None,
                          log_dir: str = "ant_logs") -> Dict[str, Any]:
    """创建默认环境配置
    
    Args:
        num_agents: 蚂蚁数量
        port: 通信端口
        executable_path: Unity可执行文件路径（None表示编辑器模式）
        log_dir: 日志目录
        
    Returns:
        配置字典
    """
    return {
        "ip": "127.0.0.1",
        "port": port,
        "numAgents": num_agents,
        "maxSteps": 1500,
        "timeout": 30,
        "executablePath": executable_path,
        "log_dir": log_dir,
        "bg": False,
        "batchmode": False,
        "nographics": False,
    }


# 保留向后兼容性
create_ant_env = AntForagingEnv
