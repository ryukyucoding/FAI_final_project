#!/usr/bin/env python3
"""
PPO Poker Agent Training Script
Based on the paper: "Probability-Based Agent for Texas Hold'em Poker"
"""

import sys
import time
from agents.agent import PPOPokerAgent  # Your PPO agent
from agents.call_player import CallPlayer
from agents.random_player import RandomPlayer
from agents.console_player import ConsolePlayer
# Import other baseline agents as needed

def create_baseline_agent(baseline_name="baseline5"):
    """Create baseline agents for training"""
    baselines = {
        "baseline0": CallPlayer(),  # Simple call player
        "baseline1": RandomPlayer(),  # Random player
        "baseline5": RandomPlayer(),  # Placeholder for strongest baseline
        "random": RandomPlayer(),
        "call": CallPlayer()
    }
    
    # You can replace these with actual baseline implementations
    return baselines.get(baseline_name, RandomPlayer())

def run_training_game(agent1, agent2, max_rounds=5, verbose=False):
    """
    Run a single training game between two agents using the actual game engine (start_poker)
    """
    from game.game import setup_config, start_poker

    # Setup config for the game
    config = setup_config(max_round=max_rounds, initial_stack=1000, small_blind_amount=5)
    config.register_player(name="player1", algorithm=agent1)
    config.register_player(name="player2", algorithm=agent2)

    # Run the game
    result = start_poker(config, verbose=1 if verbose else 0)

    # Determine winner (player with highest stack)
    players = result["players"]
    winner = max(players, key=lambda p: p["stack"])
    return {"winner": winner["name"], "rounds": max_rounds}

def train_stage_1(agent, sessions=10, games_per_session=10):
    """Train Stage 1: Against baseline agents with 5-round games"""
    print("=" * 50)
    print("STAGE 1 TRAINING: Against Baseline Agents")
    print("=" * 50)
    print(f"Sessions: {sessions}")
    print(f"Games per session: {games_per_session}")
    print(f"Rounds per game: 5")
    print()
    
    baseline_agent = create_baseline_agent("baseline5")
    
    for session in range(sessions):
        session_start = time.time()
        print(f"Session {session + 1}/{sessions}")
        
        wins = 0
        for game in range(games_per_session):
            # Run 5-round game against baseline
            result = run_training_game(agent, baseline_agent, max_rounds=5)
            if result["winner"] == "player1":
                wins += 1
        
        win_rate = wins / games_per_session
        session_time = time.time() - session_start
        
        print(f"  Win rate: {win_rate:.2f} ({wins}/{games_per_session})")
        print(f"  Session time: {session_time:.1f}s")
        print(f"  Agent stage: {agent.stage}")
        print()
        
        # Agent will automatically update its weights through receive_round_result_message
        
    print("Stage 1 completed! Transitioning to Stage 2...")
    print()

def train_stage_2(agent, sessions=40, games_per_session=10):
    """Train Stage 2: Self-play with 20-round games"""
    print("=" * 50)
    print("STAGE 2 TRAINING: Self-Play")
    print("=" * 50)
    print(f"Sessions: {sessions}")
    print(f"Games per session: {games_per_session}")
    print(f"Rounds per game: 20")
    print()
    
    for session in range(sessions):
        session_start = time.time()
        print(f"Session {session + 1}/{sessions}")
        
        # Create opponent from old versions
        if len(agent.old_agents) > 0:
            # In practice, you would create an agent with old weights
            opponent = create_baseline_agent("random")  # Placeholder
        else:
            opponent = create_baseline_agent("random")  # Fallback
        
        wins = 0
        for game in range(games_per_session):
            # Run 20-round game for self-play
            result = run_training_game(agent, opponent, max_rounds=20)
            if result["winner"] == "player1":
                wins += 1
        
        win_rate = wins / games_per_session
        session_time = time.time() - session_start
        
        print(f"  Win rate: {win_rate:.2f} ({wins}/{games_per_session})")
        print(f"  Session time: {session_time:.1f}s")
        print(f"  Old agents saved: {len(agent.old_agents)}")
        
        # Save progress every 5 sessions
        if (session + 1) % 5 == 0:
            print(f"  ✓ Model saved at session {session + 1}")
        
        print()
    
    print("Stage 2 completed! Training finished.")

def main():
    """Main training function"""
    print("PPO Poker Agent Training")
    print("Based on: Probability-Based Agent for Texas Hold'em Poker")
    print()
    
    # Create agent
    agent = PPOPokerAgent(training=True)
    print(f"Agent initialized - Stage: {agent.stage}, Sessions completed: {agent.sessions_completed}")
    print()
    
    # Check if we need to resume training
    if agent.stage == 1:
        # Start Stage 1 training
        train_stage_1(agent, sessions=10)
        
        # Agent should automatically transition to stage 2
        if agent.stage == 2:
            train_stage_2(agent, sessions=40)
    elif agent.stage == 2:
        # Resume Stage 2 training
        remaining_sessions = max(0, 40 - agent.sessions_completed)
        if remaining_sessions > 0:
            print(f"Resuming Stage 2 training - {remaining_sessions} sessions remaining")
            train_stage_2(agent, sessions=remaining_sessions)
        else:
            print("Training already completed!")
    
    print("=" * 50)
    print("TRAINING COMPLETED!")
    print("=" * 50)
    print(f"Final stage: {agent.stage}")
    print(f"Total sessions: {agent.sessions_completed}")
    print(f"Model saved at: {agent.model_path}")
    print()
    print("You can now use the trained agent in actual games!")

def test_agent():
    """Test the trained agent against baselines"""
    print("Testing trained agent...")
    
    # Load trained agent
    agent = PPOPokerAgent(training=False)  # Set training=False for testing
    
    baselines = ["baseline0", "baseline1", "baseline5", "random", "call"]
    
    for baseline_name in baselines:
        baseline = create_baseline_agent(baseline_name)
        wins = 0
        total_games = 100
        
        print(f"Testing against {baseline_name}...")
        for game in range(total_games):
            result = run_training_game(agent, baseline, max_rounds=20, verbose=False)
            if result["winner"] == "player1":
                wins += 1
        
        win_rate = wins / total_games
        print(f"  Win rate vs {baseline_name}: {win_rate:.2f} ({wins}/{total_games})")
    
    print("Testing completed!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_agent()
    else:
        main()