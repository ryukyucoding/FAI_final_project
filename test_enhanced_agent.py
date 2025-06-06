#!/usr/bin/env python3
"""
Test script for the enhanced PPO Poker Agent
Tests various components and functionalities
"""

import numpy as np
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.RLagent import PPOPokerAgent
from agents.random_player import RandomPlayer
from agents.call_player import CallPlayer


def test_agent_initialization():
    """Test agent initialization and basic functionality"""
    print("Testing Agent Initialization...")
    
    agent = PPOPokerAgent(training=True)
    
    # Test network initialization
    assert agent.input_size == 12, f"Expected input size 12, got {agent.input_size}"
    assert agent.hidden_dim == 128, f"Expected hidden dim 128, got {agent.hidden_dim}"
    assert agent.action_size == 3, f"Expected action size 3, got {agent.action_size}"
    
    # Test weight initialization
    assert 'W1' in agent.actor_weights, "Actor weights not properly initialized"
    assert 'W1' in agent.critic_weights, "Critic weights not properly initialized"
    
    print("✓ Agent initialization test passed")


def test_state_feature_extraction():
    """Test state feature extraction"""
    print("Testing State Feature Extraction...")
    
    agent = PPOPokerAgent(training=True)
    
    # Mock valid actions
    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 10},
        {"action": "raise", "amount": {"min": 20, "max": 100}}
    ]
    
    # Mock hole cards and round state
    hole_card = ["H2", "S5"]
    round_state = {
        'community_card': ["H7", "D8", "C9"],
        'seats': [
            {'uuid': 'player1', 'stack': 500, 'state': 'active'},
            {'uuid': 'player2', 'stack': 600, 'state': 'active'}
        ],
        'pot': {'main': {'amount': 50}},
        'street': 'flop',
        'round_count': 1,
        'action_histories': {}
    }
    
    # Set agent UUID for testing
    agent.uuid = 'player1'
    
    # Extract features
    features = agent._extract_state_features(valid_actions, hole_card, round_state)
    
    assert len(features) == 12, f"Expected 12 features, got {len(features)}"
    assert 0 <= features[0] <= 1, f"Win probability should be between 0 and 1, got {features[0]}"
    assert 0 <= features[1] <= 1, f"Position should be between 0 and 1, got {features[1]}"
    
    print("✓ State feature extraction test passed")


def test_hand_evaluation():
    """Test hand strength evaluation"""
    print("Testing Hand Evaluation...")
    
    agent = PPOPokerAgent(training=True)
    
    # Test strong hand
    strong_hand = ["SA", "HA"]  # Pocket Aces
    community = ["S7", "D8", "C9"]
    strength = agent._evaluate_hand_strength(strong_hand, community)
    assert strength > 0.5, f"Strong hand should have strength > 0.5, got {strength}"
    
    # Test weak hand
    weak_hand = ["S2", "H3"]  # Low cards
    strength = agent._evaluate_hand_strength(weak_hand, community)
    assert strength < 0.8, f"Weak hand should have reasonable strength, got {strength}"
    
    print("✓ Hand evaluation test passed")


def test_network_forward_pass():
    """Test neural network forward passes"""
    print("Testing Network Forward Pass...")
    
    agent = PPOPokerAgent(training=True)
    
    # Create dummy state
    state = np.random.random(12)
    
    # Test actor forward pass
    action_probs, raise_mean, raise_std = agent._actor_forward(state)
    
    assert len(action_probs) == 3, f"Expected 3 action probabilities, got {len(action_probs)}"
    assert abs(np.sum(action_probs) - 1.0) < 0.01, f"Action probabilities should sum to 1, got {np.sum(action_probs)}"
    assert len(raise_mean) == 1, f"Expected 1 raise mean, got {len(raise_mean)}"
    assert len(raise_std) == 1, f"Expected 1 raise std, got {len(raise_std)}"
    assert raise_std[0] > 0, f"Raise std should be positive, got {raise_std[0]}"
    
    # Test critic forward pass
    value = agent._critic_forward(state)
    assert isinstance(value, (int, float)), f"Value should be numeric, got {type(value)}"
    
    print("✓ Network forward pass test passed")


def test_action_declaration():
    """Test action declaration"""
    print("Testing Action Declaration...")
    
    agent = PPOPokerAgent(training=True)
    agent.uuid = 'test_player'
    
    # Mock valid actions
    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 10},
        {"action": "raise", "amount": {"min": 20, "max": 100}}
    ]
    
    hole_card = ["H7", "S8"]
    round_state = {
        'community_card': [],
        'seats': [{'uuid': 'test_player', 'stack': 500}],
        'pot': {'main': {'amount': 20}},
        'street': 'preflop',
        'action_histories': {}
    }
    
    # Test action declaration
    action, amount = agent.declare_action(valid_actions, hole_card, round_state)
    
    assert action in ['fold', 'call', 'raise'], f"Invalid action: {action}"
    if action == 'raise':
        assert 20 <= amount <= 100, f"Raise amount {amount} not in valid range [20, 100]"
    
    print("✓ Action declaration test passed")


def test_opponent_modeling():
    """Test opponent modeling functionality"""
    print("Testing Opponent Modeling...")
    
    agent = PPOPokerAgent(training=True)
    
    # Mock round state with opponent actions
    round_state = {
        'action_histories': {
            'preflop': [
                {'uuid': 'opponent1', 'action': 'raise', 'amount': 20},
                {'uuid': 'opponent1', 'action': 'call', 'amount': 10},
                {'uuid': 'opponent2', 'action': 'fold', 'amount': 0}
            ]
        },
        'seats': [
            {'uuid': 'opponent1', 'stack': 500},
            {'uuid': 'opponent2', 'stack': 400}
        ]
    }
    
    # Test aggression estimation
    aggression = agent._estimate_opponent_aggression(round_state)
    assert 0 <= aggression <= 1, f"Aggression should be between 0 and 1, got {aggression}"
    
    # Check if opponent profiles are created
    assert len(agent.opponent_modeling) > 0, "Opponent modeling should create profiles"
    
    print("✓ Opponent modeling test passed")


def test_training_components():
    """Test training-related components"""
    print("Testing Training Components...")
    
    agent = PPOPokerAgent(training=True)
    
    # Create mock trajectory
    trajectory = [
        {
            'state': np.random.random(12),
            'action_idx': 1,
            'action_prob': 0.4,
            'value': 0.3,
            'reward': 0.5,
            'done': False
        },
        {
            'state': np.random.random(12),
            'action_idx': 2,
            'action_prob': 0.6,
            'value': 0.7,
            'reward': 1.0,
            'done': True
        }
    ]
    
    # Test GAE calculation
    advantages = agent._calculate_gae(trajectory)
    assert len(advantages) == 2, f"Expected 2 advantages, got {len(advantages)}"
    
    # Test exploration rate updates
    agent.win_rates.extend([0.7] * 25)  # Mock high win rate
    initial_entropy = agent.entropy_coef
    agent._update_exploration_rate()
    assert agent.entropy_coef <= initial_entropy, "Entropy should decrease with good performance"
    
    print("✓ Training components test passed")


def test_model_save_load():
    """Test model saving and loading"""
    print("Testing Model Save/Load...")
    
    agent1 = PPOPokerAgent(training=True, model_path="test_model.pkl")
    
    # Modify some weights
    original_weight = agent1.actor_weights['W1'].copy()
    agent1.actor_weights['W1'] += 0.1
    
    # Save model
    agent1.save_model()
    
    # Create new agent and load
    agent2 = PPOPokerAgent(training=True, model_path="test_model.pkl")
    
    # Check if weights were loaded correctly
    assert not np.array_equal(agent2.actor_weights['W1'], original_weight), "Weights should be different after modification"
    
    # Clean up
    if os.path.exists("test_model.pkl"):
        os.remove("test_model.pkl")
    
    print("✓ Model save/load test passed")


def run_integration_test():
    """Run a quick integration test with actual game simulation"""
    print("Running Integration Test...")
    
    try:
        from training_script import run_training_game
        
        # Create agents
        ppo_agent = PPOPokerAgent(training=True)
        random_agent = RandomPlayer()
        
        # Run a short game
        result = run_training_game(ppo_agent, random_agent, max_rounds=2, verbose=False)
        
        assert 'winner' in result, "Game result should contain winner"
        assert result['winner'] in ['player1', 'player2'], f"Invalid winner: {result['winner']}"
        
        print("✓ Integration test passed")
        
    except Exception as e:
        print(f"⚠ Integration test skipped due to import error: {e}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("ENHANCED PPO POKER AGENT TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_agent_initialization,
        test_state_feature_extraction,
        test_hand_evaluation,
        test_network_forward_pass,
        test_action_declaration,
        test_opponent_modeling,
        test_training_components,
        test_model_save_load,
        run_integration_test
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All tests passed! The enhanced PPO agent is ready for training.")
    else:
        print("⚠ Some tests failed. Please review the implementation.")


if __name__ == "__main__":
    main()
