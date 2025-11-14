"""
蚂蚁觅食环境测试脚本

验证环境的基本功能：
- 环境创建与通信（reset/step）
- 随机动作测试
- 每秒请求数（RPS）监控
"""
import sys
import os
import time
import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 确保可以导入项目模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from GymlikeEnvWrapper.ant_env import AntForagingEnv, create_default_config


class RPSMonitor:
    """每秒请求数监控器"""
    
    def __init__(self, window_seconds: int = 5):
        """初始化RPS监控器
        
        Args:
            window_seconds: 统计窗口大小（秒）
        """
        self.window = max(1, int(window_seconds))
        self.timestamps = []
        self.last_print = time.time()

    def tick(self) -> None:
        """记录一次请求"""
        now = time.time()
        self.timestamps.append(now)
        
        # 清理过期时间戳
        cutoff = now - self.window
        self.timestamps = [ts for ts in self.timestamps if ts >= cutoff]
        
        # 定期打印RPS
        if now - self.last_print >= self.window:
            rps = len(self.timestamps) / float(self.window)
            print(f"RPS({self.window}s) = {rps:.2f}/s")
            self.last_print = now


def generate_random_actions(num_agents: int) -> np.ndarray:
    """生成随机动作
    
    Args:
        num_agents: 智能体数量
        
    Returns:
        动作数组
    """
    return np.random.randint(0, 3, size=(num_agents * 2,), dtype=np.int32)


def run(env_port: int = 5555, 
        num_agents: int = 1024, 
        executable_path: str = None, 
        rps_window: int = 5) -> None:
    """运行环境测试
    
    Args:
        env_port: 环境端口
        num_agents: 智能体数量
        executable_path: Unity可执行文件路径
        rps_window: RPS统计窗口（秒）
    """
    print("🧪 环境测试 + RPS监控")
    
    # 创建环境
    config = create_default_config(
        num_agents=num_agents, 
        port=env_port, 
        executable_path=executable_path, 
        log_dir="test_ant_logs"
    )
    env = AntForagingEnv(config)
    
    # 重置环境
    obs, info = env.reset()
    print(f"✅ 重置成功，智能体数量: {len(obs)}")

    # 启动RPS监控
    rps = RPSMonitor(window_seconds=rps_window)

    try:
        while True:
            actions = generate_random_actions(env.numAgents)
            obs, rewards, terminateds, truncateds, info = env.step(actions)
            rps.tick()
            
            current_step = info.get('current_step', 0)
            if current_step % 100 == 0:
                print(f"Step: {current_step}")
            
            if info.get('envDone', False):
                episode = info.get('current_episode', 0)
                print(f"Episode {episode} 完成")
                
    except KeyboardInterrupt:
        print("\n⏹️ 测试中断")
    finally:
        env.close()
        print("✅ 测试结束")


if __name__ == "__main__":
    exe_path = r'D:\Program Files (x86)\VScodeWorkSpace\GameEnvUnity\AntRL\AntEnv\Ant.exe'
    run(env_port=5555, num_agents=512, executable_path=exe_path, rps_window=5)
