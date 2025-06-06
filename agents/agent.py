import numpy as np
import random
from collections import deque
import pickle
import os
from game.players import BasePokerPlayer


class PPOPokerAgent(BasePokerPlayer):
    def __init__(self, training=True, model_path="ppo_poker_model.pkl"):
        # Training parameters
        self.training = training
        self.model_path = model_path
        
        # PPO hyperparameters
        self.learning_rate = 0.001
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
        # Network architecture (4-layer MLP with hidden dimension 64)
        self.input_size = 8  # State space as described in paper
        self.hidden_dim = 64
        self.action_size = 3  # fold, call, raise probabilities
        
        # Initialize networks (actor and critic)
        self.actor_weights = self._initialize_actor_weights()
        self.critic_weights = self._initialize_critic_weights()
        
        # Training data collection
        self.trajectories = []
        self.current_trajectory = []
        self.epochs_per_update = 20
        self.trajectories_per_session = 10
        
        # Stage tracking (two-stage training)
        self.stage = 1  # 1 for first stage (5-round games), 2 for second stage (self-play)
        self.sessions_completed = 0
        self.rounds_in_current_game = 0
        
        # Old agents for self-play (stage 2)
        self.old_agents = deque(maxlen=5)  # Window size of 5
        self.save_interval = 5  # Save model every 5 sessions
        
        # Game state tracking
        self.current_state = None
        self.last_action = None
        self.last_action_prob = None
        self.game_start_stack = 0
        
        # Monte Carlo simulation for round rate
        self.mc_simulations = 1000
        
        # Load existing model if available
        self.load_model()
    
    def _initialize_actor_weights(self):
        """Initialize actor network weights (policy network)"""
        weights = {}
        # Input to hidden
        weights['W1'] = np.random.randn(self.input_size, self.hidden_dim) * 0.1
        weights['b1'] = np.zeros((1, self.hidden_dim))
        # Hidden layers
        weights['W2'] = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1
        weights['b2'] = np.zeros((1, self.hidden_dim))
        weights['W3'] = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1
        weights['b3'] = np.zeros((1, self.hidden_dim))
        # Output layer - action probabilities
        weights['W4_action'] = np.random.randn(self.hidden_dim, self.action_size) * 0.1
        weights['b4_action'] = np.zeros((1, self.action_size))
        # Output layer - raise amount (mean and std)
        weights['W4_raise_mean'] = np.random.randn(self.hidden_dim, 1) * 0.1
        weights['b4_raise_mean'] = np.zeros((1, 1))
        weights['W4_raise_std'] = np.random.randn(self.hidden_dim, 1) * 0.1
        weights['b4_raise_std'] = np.zeros((1, 1))
        return weights
    
    def _initialize_critic_weights(self):
        """Initialize critic network weights (value network)"""
        weights = {}
        # Input to hidden
        weights['W1'] = np.random.randn(self.input_size, self.hidden_dim) * 0.1
        weights['b1'] = np.zeros((1, self.hidden_dim))
        # Hidden layers
        weights['W2'] = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1
        weights['b2'] = np.zeros((1, self.hidden_dim))
        weights['W3'] = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1
        weights['b3'] = np.zeros((1, self.hidden_dim))
        # Output layer - state value
        weights['W4'] = np.random.randn(self.hidden_dim, 1) * 0.1
        weights['b4'] = np.zeros((1, 1))
        return weights
    
    def _actor_forward(self, state):
        """Forward pass through actor network"""
        x = state.reshape(1, -1)
        
        # Hidden layers with ReLU activation
        z1 = np.dot(x, self.actor_weights['W1']) + self.actor_weights['b1']
        a1 = np.maximum(0, z1)  # ReLU
        
        z2 = np.dot(a1, self.actor_weights['W2']) + self.actor_weights['b2']
        a2 = np.maximum(0, z2)  # ReLU
        
        z3 = np.dot(a2, self.actor_weights['W3']) + self.actor_weights['b3']
        a3 = np.maximum(0, z3)  # ReLU
        
        # Action probabilities (softmax)
        action_logits = np.dot(a3, self.actor_weights['W4_action']) + self.actor_weights['b4_action']
        action_probs = self._softmax(action_logits)
        
        # Raise amount distribution parameters
        raise_mean = np.dot(a3, self.actor_weights['W4_raise_mean']) + self.actor_weights['b4_raise_mean']
        raise_std_logit = np.dot(a3, self.actor_weights['W4_raise_std']) + self.actor_weights['b4_raise_std']
        raise_std = np.exp(raise_std_logit)  # Ensure positive std
        
        return action_probs.flatten(), raise_mean.flatten(), raise_std.flatten()
    
    def _critic_forward(self, state):
        """Forward pass through critic network"""
        x = state.reshape(1, -1)
        
        # Hidden layers with ReLU activation
        z1 = np.dot(x, self.critic_weights['W1']) + self.critic_weights['b1']
        a1 = np.maximum(0, z1)  # ReLU
        
        z2 = np.dot(a1, self.critic_weights['W2']) + self.critic_weights['b2']
        a2 = np.maximum(0, z2)  # ReLU
        
        z3 = np.dot(a2, self.critic_weights['W3']) + self.critic_weights['b3']
        a3 = np.maximum(0, z3)  # ReLU
        
        # State value
        value = np.dot(a3, self.critic_weights['W4']) + self.critic_weights['b4']
        
        return value.flatten()[0]
    
    def _softmax(self, x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def _compute_round_rate(self, hole_card, community_cards):
        """Compute round rate using Monte Carlo simulation"""
        if not hole_card:
            return 0.5
        
        wins = 0
        for _ in range(self.mc_simulations):
            # Simulate random opponent cards and remaining community cards
            # This is a simplified simulation
            opponent_strength = random.random()
            our_strength = self._evaluate_hand_strength(hole_card, community_cards)
            
            if our_strength > opponent_strength:
                wins += 1
        
        return wins / self.mc_simulations
    
    def _evaluate_hand_strength(self, hole_card, community_cards):
        """Simplified hand strength evaluation"""
        # This is a placeholder for actual poker hand evaluation
        # In practice, you would implement proper poker hand rankings
        if not hole_card:
            return 0.0
        
        # Simple strength based on high cards
        strength = 0.0
        for card in hole_card:
            rank = card[1]
            if rank == 'A':
                strength += 0.4
            elif rank == 'K':
                strength += 0.3
            elif rank == 'Q':
                strength += 0.2
            elif rank == 'J':
                strength += 0.1
            elif rank == 'T':
                strength += 0.05
        
        # Add community card considerations
        for card in community_cards:
            rank = card[1]
            if rank in ['A', 'K', 'Q']:
                strength += 0.05
        
        return min(strength, 1.0)
    
    def _extract_state_features(self, valid_actions, hole_card, round_state):
        """Extract state features as described in the paper"""
        features = np.zeros(self.input_size)
        
        if not round_state:
            return features
        
        # 1. Round rate (computed by Monte Carlo)
        community_cards = round_state.get('community_card', [])
        features[0] = self._compute_round_rate(hole_card, community_cards)
        
        # 2. First player to act (boolean)
        seats = round_state.get('seats', [])
        my_uuid = getattr(self, 'uuid', None)
        first_to_act = True  # Simplified - assume we're first if unsure
        for i, seat in enumerate(seats):
            if seat.get('uuid') == my_uuid:
                first_to_act = (i == 0)
                break
        features[1] = 1.0 if first_to_act else 0.0
        
        # 3. Net profit if fold
        fold_loss = valid_actions[0].get('amount', 0) if len(valid_actions) > 0 else 0
        features[2] = -fold_loss / 100.0  # Normalized
        
        # 4. Net profit if call
        call_cost = valid_actions[1].get('amount', 0) if len(valid_actions) > 1 else 0
        pot_size = round_state.get('pot', {}).get('main', {}).get('size', 0)
        expected_call_profit = features[0] * pot_size - (1 - features[0]) * call_cost
        features[3] = expected_call_profit / 100.0  # Normalized
        
        # 5. Net profit if raise (minimum)
        if len(valid_actions) > 2 and valid_actions[2].get('amount', {}).get('min', -1) != -1:
            min_raise = valid_actions[2]['amount']['min']
            expected_raise_profit = features[0] * (pot_size + min_raise) - (1 - features[0]) * min_raise
            features[4] = expected_raise_profit / 100.0  # Normalized
        else:
            features[4] = features[3]  # Same as call if raise not available
        
        # 6. Number of remaining rounds
        total_rounds = 20 if self.stage == 2 else 5
        current_round = round_state.get('round_count', 1)
        features[5] = (total_rounds - current_round) / total_rounds
        
        # 7. Current street (0=preflop, 1=flop, 2=turn, 3=river)
        street_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
        street = round_state.get('street', 'preflop')
        features[6] = street_map.get(street, 0) / 3.0
        
        # 8. Stack size (smaller means more aggressive)
        my_stack = 0
        for seat in seats:
            if seat.get('uuid') == my_uuid:
                my_stack = seat.get('stack', 0)
                break
        features[7] = my_stack / 1000.0  # Normalized
        
        return features
    
    def declare_action(self, valid_actions, hole_card, round_state):
        """Main decision-making function using PPO"""
        try:
            # Extract state features
            state = self._extract_state_features(valid_actions, hole_card, round_state)
            self.current_state = state
            
            # Get action probabilities and raise parameters from actor network
            action_probs, raise_mean, raise_std = self._actor_forward(state)
            
            # Filter valid actions - ensure we only choose from available actions
            valid_action_indices = []
            for i, action_info in enumerate(valid_actions):
                if i < 3:  # fold, call, raise
                    if action_info["action"] == "raise":
                        # Check if raise is actually valid
                        amount_info = action_info.get("amount", {})
                        if isinstance(amount_info, dict) and amount_info.get("min", -1) != -1:
                            valid_action_indices.append(i)
                    else:
                        valid_action_indices.append(i)
            
            # If no valid actions, default to call (index 1) or fold (index 0)
            if not valid_action_indices:
                valid_action_indices = [0]  # fold as last resort
            
            # Normalize probabilities for valid actions only
            valid_probs = np.array([action_probs[i] if i < len(action_probs) else 0.1 for i in valid_action_indices])
            valid_probs = valid_probs / np.sum(valid_probs)
            
            # Sample action based on valid probabilities
            chosen_idx = np.random.choice(len(valid_action_indices), p=valid_probs)
            action_idx = valid_action_indices[chosen_idx]
            
            # Ensure action_idx is within bounds
            action_idx = min(action_idx, len(valid_actions) - 1)
            
            action_info = valid_actions[action_idx]
            action = action_info["action"]
            amount = action_info["amount"]
            
            # Handle raise amount using continuous distribution
            if action == "raise" and isinstance(amount, dict):
                min_raise = amount.get("min", -1)
                max_raise = amount.get("max", -1)
                
                if min_raise != -1 and max_raise != -1 and min_raise <= max_raise:
                    # Sample from normal distribution and transform to valid range
                    sampled_amount = np.random.normal(raise_mean[0], max(raise_std[0], 0.1))
                    # Transform using sigmoid to bounded range
                    normalized = 1 / (1 + np.exp(-sampled_amount))  # Sigmoid
                    amount = int(min_raise + normalized * (max_raise - min_raise))
                    amount = max(min_raise, min(max_raise, amount))
                else:
                    # Raise not valid, fall back to call
                    if len(valid_actions) > 1:
                        action_info = valid_actions[1]
                        action = action_info["action"]
                        amount = action_info["amount"]
                        action_idx = 1
                    else:
                        # Fall back to fold
                        action_info = valid_actions[0]
                        action = action_info["action"]
                        amount = action_info["amount"]
                        action_idx = 0
            
            # Store action info for training
            if self.training:
                value = self._critic_forward(state)
                actual_prob = action_probs[action_idx] if action_idx < len(action_probs) else 0.1
                self.last_action = {
                    'state': state,
                    'action_idx': action_idx,
                    'action_prob': actual_prob,
                    'value': value,
                    'action': action,
                    'amount': amount
                }
            
            return action, amount
            
        except Exception as e:
            print(f"Error in declare_action: {e}")
            # Safe fallback: always call if possible, otherwise fold
            if len(valid_actions) > 1:
                return valid_actions[1]["action"], valid_actions[1]["amount"]
            else:
                return valid_actions[0]["action"], valid_actions[0]["amount"]
    
    def receive_game_start_message(self, game_info):
        """Called when game starts"""
        self.current_trajectory = []
        self.rounds_in_current_game = 0
        # Determine number of rounds based on stage
        if self.stage == 1:
            # First stage: 5-round games
            pass
        else:
            # Second stage: 20-round games (self-play)
            pass
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        """Called when new round starts"""
        self.rounds_in_current_game += 1
        
        # Get initial stack size
        my_uuid = getattr(self, 'uuid', None)
        if my_uuid:
            for seat in seats:
                if seat.get('uuid') == my_uuid:
                    if round_count == 1:
                        self.game_start_stack = seat.get('stack', 0)
                    break
    
    def receive_street_start_message(self, street, round_state):
        """Called when new street starts"""
        pass
    
    def receive_game_update_message(self, action, round_state):
        """Called when game state updates"""
        pass
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        """Called when round ends"""
        if not self.training or not self.last_action:
            return
        
        # Calculate reward
        reward = self._calculate_reward(winners, round_state)
        
        # Add experience to current trajectory
        experience = {
            'state': self.last_action['state'],
            'action_idx': self.last_action['action_idx'],
            'action_prob': self.last_action['action_prob'],
            'value': self.last_action['value'],
            'reward': reward,
            'done': self._is_game_over(round_state)
        }
        self.current_trajectory.append(experience)
        
        # If game is over, finish trajectory and possibly train
        if experience['done']:
            self.trajectories.append(self.current_trajectory.copy())
            self.current_trajectory = []
            
            # Train when we have enough trajectories
            if len(self.trajectories) >= self.trajectories_per_session:
                self._train_ppo()
                self.trajectories = []
                self.sessions_completed += 1
                
                # Stage transition
                if self.stage == 1 and self.sessions_completed >= 10:
                    self._transition_to_stage_2()
                
                # Save model periodically
                if self.sessions_completed % self.save_interval == 0:
                    self._save_old_agent()
                    self.save_model()
    
    def _calculate_reward(self, winners, round_state):
        """Calculate reward for PPO training"""
        reward = 0.0
        my_uuid = getattr(self, 'uuid', None)
        
        # Check if we won
        if my_uuid:
            for winner in winners:
                if winner.get('uuid') == my_uuid:
                    reward += 1.0  # Win reward
                    break
            else:
                reward -= 0.1  # Small loss penalty
        
        return reward
    
    def _is_game_over(self, round_state):
        """Check if the game is over"""
        # Game ends when we reach the round limit or someone runs out of money
        if not round_state:
            return True
            
        # Check if any player is eliminated (stack = 0)
        seats = round_state.get('seats', [])
        active_players = 0
        for seat in seats:
            if seat.get('stack', 0) > 0:
                active_players += 1
        
        if active_players <= 1:
            return True
            
        # Check round limits
        current_round = round_state.get('round_count', 1)
        if self.stage == 1:
            return current_round >= 5
        else:
            return current_round >= 20
    
    def _train_ppo(self):
        """Train the PPO agent"""
        if not self.trajectories:
            return
        
        # Prepare training data
        all_states = []
        all_actions = []
        all_old_probs = []
        all_rewards = []
        all_values = []
        all_advantages = []
        
        for trajectory in self.trajectories:
            # Calculate advantages using GAE
            advantages = self._calculate_gae(trajectory)
            
            for i, exp in enumerate(trajectory):
                all_states.append(exp['state'])
                all_actions.append(exp['action_idx'])
                all_old_probs.append(exp['action_prob'])
                all_rewards.append(exp['reward'])
                all_values.append(exp['value'])
                all_advantages.append(advantages[i])
        
        # Convert to numpy arrays
        all_states = np.array(all_states)
        all_actions = np.array(all_actions)
        all_old_probs = np.array(all_old_probs)
        all_advantages = np.array(all_advantages)
        all_returns = all_advantages + np.array(all_values)
        
        # Normalize advantages
        all_advantages = (all_advantages - np.mean(all_advantages)) / (np.std(all_advantages) + 1e-8)
        
        # PPO training epochs
        for epoch in range(self.epochs_per_update):
            # This is a simplified PPO update
            # In practice, you would implement proper gradient computation and backpropagation
            self._simplified_ppo_update(all_states, all_actions, all_old_probs, all_advantages, all_returns)
    
    def _calculate_gae(self, trajectory):
        """Calculate Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(trajectory))):
            exp = trajectory[i]
            if i == len(trajectory) - 1:
                next_value = 0  # Terminal state
            else:
                next_value = trajectory[i + 1]['value']
            
            delta = exp['reward'] + self.gamma * next_value - exp['value']
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def _simplified_ppo_update(self, states, actions, old_probs, advantages, returns):
        """Simplified PPO update (placeholder for full implementation)"""
        # This is a placeholder for proper PPO gradient computation
        # In practice, you would implement full backpropagation with clipping
        learning_signal = np.mean(advantages) * self.learning_rate
        
        # Simple weight updates
        for key in self.actor_weights:
            if 'W' in key:
                self.actor_weights[key] += learning_signal * 0.0001 * np.random.randn(*self.actor_weights[key].shape)
        
        for key in self.critic_weights:
            if 'W' in key:
                self.critic_weights[key] += learning_signal * 0.0001 * np.random.randn(*self.critic_weights[key].shape)
    
    def _transition_to_stage_2(self):
        """Transition from stage 1 to stage 2 (self-play)"""
        print("Transitioning to Stage 2: Self-play with 20-round games")
        self.stage = 2
        self.sessions_completed = 0
        # Save current agent as first opponent for self-play
        self._save_old_agent()
    
    def _save_old_agent(self):
        """Save current agent weights for self-play opponents"""
        agent_copy = {
            'actor_weights': {k: v.copy() for k, v in self.actor_weights.items()},
            'critic_weights': {k: v.copy() for k, v in self.critic_weights.items()}
        }
        self.old_agents.append(agent_copy)
    
    def save_model(self):
        """Save the trained model"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'actor_weights': self.actor_weights,
                    'critic_weights': self.critic_weights,
                    'stage': self.stage,
                    'sessions_completed': self.sessions_completed,
                    'old_agents': list(self.old_agents)
                }, f)
            print(f"Model saved: Stage {self.stage}, Sessions {self.sessions_completed}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self):
        """Load a previously trained model"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.actor_weights = data['actor_weights']
                    self.critic_weights = data['critic_weights']
                    self.stage = data.get('stage', 1)
                    self.sessions_completed = data.get('sessions_completed', 0)
                    self.old_agents = deque(data.get('old_agents', []), maxlen=5)
                print(f"Loaded model: Stage {self.stage}, Sessions {self.sessions_completed}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting with fresh model")


def setup_ai():
    """Setup function required by the framework"""
    return PPOPokerAgent(training=True)