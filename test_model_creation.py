#!/usr/bin/env python3
"""
測試PPO代理模型創建和保存
"""

import os
import sys
import pickle
from datetime import datetime

# 添加agents目錄到Python路徑
sys.path.append('agents')

def test_model_creation():
    """測試模型創建和保存"""
    print("Testing PPO Poker Agent Model Creation")
    print("=" * 50)
    
    try:
        # 導入你的代理 - 根據你的實際文件名調整
        try:
            from agents import PPOPokerAgent  # 如果文件名是 agent.py
        except ImportError:
                # 嘗試直接導入
                sys.path.append('.')
                from agents import PPOPokerAgent
        
        # 創建代理實例
        print("1. Creating PPO agent...")
        agent = PPOPokerAgent(training=True, model_path="ppo_poker_model.pkl")
        print(f"   ✓ Agent created successfully")
        print(f"   - Stage: {agent.stage}")
        print(f"   - Sessions completed: {agent.sessions_completed}")
        print(f"   - Model path: {agent.model_path}")
        
        # 手動保存模型
        print("\n2. Saving model...")
        agent.save_model()
        
        # 檢查文件是否創建
        if os.path.exists("ppo_poker_model.pkl"):
            file_size = os.path.getsize("ppo_poker_model.pkl")
            print(f"   ✓ Model file created: ppo_poker_model.pkl ({file_size} bytes)")
        else:
            print("   ✗ Model file not created")
            return False
        
        # 測試加載模型
        print("\n3. Testing model loading...")
        agent2 = PPOPokerAgent(training=False, model_path="ppo_poker_model.pkl")
        print(f"   ✓ Model loaded successfully")
        print(f"   - Stage: {agent2.stage}")
        print(f"   - Sessions completed: {agent2.sessions_completed}")
        
        # 檢查模型內容
        print("\n4. Checking model contents...")
        with open("ppo_poker_model.pkl", 'rb') as f:
            data = pickle.load(f)
        
        print(f"   - Actor weights keys: {list(data['actor_weights'].keys())[:3]}...")
        print(f"   - Critic weights keys: {list(data['critic_weights'].keys())[:3]}...")
        print(f"   - Stage: {data['stage']}")
        print(f"   - Sessions: {data['sessions_completed']}")
        print(f"   - Old agents: {len(data['old_agents'])}")
        
        print("\n" + "=" * 50)
        print("✓ All tests passed! Model creation working correctly.")
        return True
        
    except ImportError as e:
        print(f"✗ Cannot import agent: {e}")
        print("Please check:")
        print("1. Your agent file is named correctly")
        print("2. The file is in the agents/ directory")
        print("3. The class name is PPOPokerAgent")
        return False
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def show_directory_structure():
    """顯示當前目錄結構"""
    print("\nCurrent directory structure:")
    print("=" * 30)
    
    for root, dirs, files in os.walk('.'):
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith(('.py', '.pkl', '.txt', '.md')):
                print(f"{subindent}{file}")
        
        if level > 2:  # 限制深度
            break

def create_test_agent():
    """創建一個簡單的測試代理來驗證保存功能"""
    print("Creating minimal test agent...")
    
    try:
        # 直接創建一個最小的代理來測試保存
        import numpy as np
        
        test_data = {
            'actor_weights': {
                'W1': np.random.randn(8, 64) * 0.1,
                'b1': np.zeros((1, 64))
            },
            'critic_weights': {
                'W1': np.random.randn(8, 64) * 0.1,
                'b1': np.zeros((1, 64))
            },
            'stage': 1,
            'sessions_completed': 0,
            'old_agents': []
        }
        
        # 保存測試數據
        with open('test_ppo_model.pkl', 'wb') as f:
            pickle.dump(test_data, f)
        
        print("✓ Test model created: test_ppo_model.pkl")
        
        # 驗證加載
        with open('test_ppo_model.pkl', 'rb') as f:
            loaded_data = pickle.load(f)
        
        print("✓ Test model loaded successfully")
        print(f"  Stage: {loaded_data['stage']}")
        return True
        
    except Exception as e:
        print(f"✗ Error creating test model: {e}")
        return False

if __name__ == "__main__":
    print(f"Running model creation test at: {datetime.now()}")
    print(f"Working directory: {os.getcwd()}")
    
    # 顯示目錄結構
    show_directory_structure()
    
    # 測試模型創建
    success = test_model_creation()
    
    if not success:
        print("\nTrying alternative test...")
        create_test_agent()
    
    print(f"\nFinal directory contents:")
    pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
    if pkl_files:
        print(f"PKL files found: {pkl_files}")
    else:
        print("No PKL files found in current directory")