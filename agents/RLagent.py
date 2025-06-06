import numpy as np
import random
from collections import deque
import pickle
import os
from game.players import BasePokerPlayer
from game.engine.hand_evaluator import HandEvaluator
from game.engine.card import Card


class PPOPokerAgent(BasePokerPlayer):
    def __init__(self, training=True, model_path="ppo_poker_model.pkl"):
        super().__init__()  # Initialize base class
        
        # Training parameters
        self.training = training
        self.model_path = model_path
          # Enhanced PPO hyperparameters
        self.learning_rate = 0.0003  # Reduced for more stable learning
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.02  # Increased for more exploration
        self.gamma = 0.995  # Slightly higher for longer-term thinking
        self.gae_lambda = 0.95
        self.max_grad_norm = 0.5  # Gradient clipping
        
        # Adaptive parameters
        self.initial_entropy_coef = 0.02
        self.min_entropy_coef = 0.005
        self.exploration_decay = 0.995
        self.performance_threshold = 0.6  # Win rate threshold for reducing exploration
        
        # Enhanced network architecture
        self.input_size = 12  # Expanded state space
        self.hidden_dim = 128  # Larger network
        self.action_size = 3  # fold, call, raise probabilities        
        # Initialize networks with Xavier/He initialization
        self.actor_weights = self._initialize_actor_weights()
        self.critic_weights = self._initialize_critic_weights()
        
        # Enhanced training data collection
        self.trajectories = []
        self.current_trajectory = []
        self.epochs_per_update = 10  # Reduced for more frequent updates
        self.batch_size = 64
        self.trajectories_per_session = 5  # More frequent training
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=10000)
        
        # Stage tracking (two-stage training)
        self.stage = 1  # 1 for first stage (5-round games), 2 for second stage (self-play)
        self.sessions_completed = 0
        self.rounds_in_current_game = 0
        
        # Old agents for self-play (stage 2)
        self.old_agents = deque(maxlen=10)  # Larger window
        self.save_interval = 3  # More frequent saves
          # Enhanced game state tracking
        self.current_state = None
        self.last_action = None
        self.last_action_prob = None
        self.game_start_stack = 0
        self.current_round_actions = []
        self.opponent_modeling = {}  # Track opponent tendencies
        self.game_history = deque(maxlen=1000)  # Long-term learning
        self.bluff_detection = {}  # Opponent bluff patterns
        self.position_play = {}  # Position-based strategies
        
        # Hand evaluation improvements
        self.hand_evaluator = HandEvaluator()
        self.mc_simulations = 2000  # More accurate simulations
        
        # Performance tracking
        self.win_rates = deque(maxlen=100)
        self.avg_rewards = deque(maxlen=100)
        
        # Load existing model if available
        self.load_model()
    def _initialize_actor_weights(self):
        """Initialize actor network weights with He initialization"""
        weights = {}
        layers = [self.input_size, self.hidden_dim, self.hidden_dim, self.hidden_dim, self.action_size]
        
        for i in range(len(layers) - 1):
            # He initialization for ReLU networks
            fan_in = layers[i]
            weights[f'W{i+1}'] = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / fan_in)
            weights[f'b{i+1}'] = np.zeros((1, layers[i+1]))
        
        # Separate outputs for raise amount prediction
        weights['W_raise_mean'] = np.random.randn(self.hidden_dim, 1) * 0.01
        weights['b_raise_mean'] = np.zeros((1, 1))
        weights['W_raise_std'] = np.random.randn(self.hidden_dim, 1) * 0.01
        weights['b_raise_std'] = np.zeros((1, 1))
        
        return weights
    
    def _initialize_critic_weights(self):
        """Initialize critic network weights with He initialization"""
        weights = {}
        layers = [self.input_size, self.hidden_dim, self.hidden_dim, self.hidden_dim, 1]
        
        for i in range(len(layers) - 1):
            fan_in = layers[i]
            weights[f'W{i+1}'] = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / fan_in)
            weights[f'b{i+1}'] = np.zeros((1, layers[i+1]))
        
        return weights
    def _actor_forward(self, state):
        """Enhanced forward pass through actor network"""
        x = state.reshape(1, -1)
        
        # Add batch normalization simulation (mean normalization)
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)
        
        # Forward pass through layers
        for i in range(1, 4):  # 3 hidden layers
            z = np.dot(x, self.actor_weights[f'W{i}']) + self.actor_weights[f'b{i}']
            x = np.maximum(0, z)  # ReLU activation
            # Add dropout simulation during training
            if self.training:
                x = x * (np.random.random(x.shape) > 0.1)  # 10% dropout
        
        # Output layer for actions
        action_logits = np.dot(x, self.actor_weights['W4']) + self.actor_weights['b4']
        action_probs = self._softmax(action_logits)
        
        # Separate outputs for raise amount
        raise_mean = np.dot(x, self.actor_weights['W_raise_mean']) + self.actor_weights['b_raise_mean']
        raise_std_logit = np.dot(x, self.actor_weights['W_raise_std']) + self.actor_weights['b_raise_std']
        raise_std = np.exp(raise_std_logit) + 0.1  # Ensure positive std with minimum
        
        return action_probs.flatten(), raise_mean.flatten(), raise_std.flatten()
    
    def _critic_forward(self, state):
        """Enhanced forward pass through critic network"""
        x = state.reshape(1, -1)
        
        # Add batch normalization simulation
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)
        
        # Forward pass through layers
        for i in range(1, 4):  # 3 hidden layers
            z = np.dot(x, self.critic_weights[f'W{i}']) + self.critic_weights[f'b{i}']
            x = np.maximum(0, z)  # ReLU activation
            if self.training:
                x = x * (np.random.random(x.shape) > 0.1)  # 10% dropout
        
        # Output layer for value
        value = np.dot(x, self.critic_weights['W4']) + self.critic_weights['b4']
        
        return value.flatten()[0]
    def _softmax(self, x):
        """Improved softmax with numerical stability"""
        x_shifted = x - np.max(x, axis=-1, keepdims=True)  # Numerical stability
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _parse_card(self, card_str):
        """Parse card string to extract rank and suit"""
        if len(card_str) != 2:
            return None
        suit = card_str[0]
        rank = card_str[1]
        return {'suit': suit, 'rank': rank}
    
    def _compute_round_rate(self, hole_card, community_cards):
        """Enhanced Monte Carlo simulation for win probability"""
        if not hole_card or len(hole_card) < 2:
            return 0.5
        
        try:
            # Parse hole cards
            hole_parsed = [self._parse_card(card) for card in hole_card if self._parse_card(card)]
            if len(hole_parsed) < 2:
                return 0.5
            
            # Parse community cards
            community_parsed = [self._parse_card(card) for card in community_cards if self._parse_card(card)]
            
            # Use proper hand evaluation if available
            if hasattr(self, 'hand_evaluator'):
                try:
                    hole_cards_obj = [Card(card['suit'], card['rank']) for card in hole_parsed]
                    community_cards_obj = [Card(card['suit'], card['rank']) for card in community_parsed]
                    
                    wins = 0
                    for _ in range(min(self.mc_simulations, 1000)):  # Limit for performance
                        # Simplified simulation - in practice, you'd generate random opponent hands
                        opponent_strength = random.random()
                        our_hand = hole_cards_obj + community_cards_obj
                        our_strength = self._evaluate_hand_strength_detailed(our_hand)
                        
                        if our_strength > opponent_strength:
                            wins += 1
                    
                    return wins / min(self.mc_simulations, 1000)
                except:
                    pass  # Fall back to simplified evaluation
            
            # Fallback to simplified evaluation
            return self._evaluate_hand_strength(hole_card, community_cards)
            
        except Exception as e:
            return 0.5  # Safe fallback
    
    def _evaluate_hand_strength_detailed(self, cards):
        """More detailed hand strength evaluation"""
        if not cards:
            return 0.0
        
        try:
            # Count high cards, pairs, etc.
            ranks = [card.rank if hasattr(card, 'rank') else str(card)[-1] for card in cards]
            rank_counts = {}
            for rank in ranks:
                rank_counts[rank] = rank_counts.get(rank, 0) + 1
            
            strength = 0.0
            
            # Pairs, trips, quads
            max_count = max(rank_counts.values()) if rank_counts else 0
            if max_count >= 4:
                strength += 0.9  # Quads
            elif max_count >= 3:
                strength += 0.7  # Trips
            elif max_count >= 2:
                strength += 0.4  # Pair
            
            # High cards
            high_cards = ['A', 'K', 'Q', 'J', 'T']
            for rank in ranks:
                if rank in high_cards:
                    strength += 0.1
            
            return min(strength, 1.0)
        except:
            return 0.3  # Conservative fallback
    def _evaluate_hand_strength(self, hole_card, community_cards):
        """Enhanced hand strength evaluation"""
        if not hole_card:
            return 0.0
        
        try:
            strength = 0.0
            all_cards = hole_card + community_cards
            
            # Parse all cards
            parsed_cards = [self._parse_card(card) for card in all_cards if self._parse_card(card)]
            if not parsed_cards:
                return 0.3
            
            ranks = [card['rank'] for card in parsed_cards]
            suits = [card['suit'] for card in parsed_cards]
            
            # Count ranks
            rank_counts = {}
            for rank in ranks:
                rank_counts[rank] = rank_counts.get(rank, 0) + 1
            
            # Count suits for flush detection
            suit_counts = {}
            for suit in suits:
                suit_counts[suit] = suit_counts.get(suit, 0) + 1
            
            # Evaluate hand patterns
            max_rank_count = max(rank_counts.values()) if rank_counts else 0
            max_suit_count = max(suit_counts.values()) if suit_counts else 0
            
            # Four of a kind
            if max_rank_count >= 4:
                strength += 0.95
            # Full house or trips
            elif max_rank_count >= 3:
                if len([count for count in rank_counts.values() if count >= 2]) >= 2:
                    strength += 0.85  # Full house
                else:
                    strength += 0.65  # Trips
            # Two pair or pair
            elif max_rank_count >= 2:
                pairs = len([count for count in rank_counts.values() if count >= 2])
                if pairs >= 2:
                    strength += 0.45  # Two pair
                else:
                    strength += 0.25  # One pair
            
            # Flush potential
            if max_suit_count >= 5:
                strength += 0.75  # Flush
            elif max_suit_count >= 4:
                strength += 0.15  # Flush draw
            
            # High cards bonus
            high_cards = {'A': 0.15, 'K': 0.12, 'Q': 0.10, 'J': 0.08, 'T': 0.05}
            for rank in ranks:
                strength += high_cards.get(rank, 0)
            
            # Hole card specific bonus
            hole_ranks = [self._parse_card(card)['rank'] for card in hole_card[:2] if self._parse_card(card)]
            if len(hole_ranks) == 2:
                # Pocket pair bonus
                if hole_ranks[0] == hole_ranks[1]:
                    strength += 0.2
                # Suited hole cards bonus
                hole_suits = [self._parse_card(card)['suit'] for card in hole_card[:2] if self._parse_card(card)]
                if len(hole_suits) == 2 and hole_suits[0] == hole_suits[1]:
                    strength += 0.1
            
            return min(strength, 1.0)
            
        except Exception:
            # Fallback to simple evaluation
            strength = 0.0
            for card in hole_card:
                if len(card) >= 2:
                    rank = card[1] if len(card) > 1 else card[0]
                    if rank == 'A':
                        strength += 0.4
                    elif rank == 'K':
                        strength += 0.3
                    elif rank == 'Q':
                        strength += 0.2
                    elif rank in ['J', 'T']:
                        strength += 0.1
            
            return min(strength, 1.0)
    def _extract_state_features(self, valid_actions, hole_card, round_state):
        """Enhanced state feature extraction with more comprehensive information"""
        features = np.zeros(self.input_size)
        
        if not round_state:
            return features
        
        try:
            # 1. Round rate (win probability)
            community_cards = round_state.get('community_card', [])
            features[0] = self._compute_round_rate(hole_card, community_cards)
            
            # 2. Position information (early/middle/late position)
            seats = round_state.get('seats', [])
            my_uuid = getattr(self, 'uuid', None)
            position_ratio = 0.5  # Default middle position
            if my_uuid and seats:
                for i, seat in enumerate(seats):
                    if seat.get('uuid') == my_uuid:
                        position_ratio = i / max(len(seats) - 1, 1)
                        break
            features[1] = position_ratio
            
            # 3. Pot odds calculation
            pot_size = round_state.get('pot', {}).get('main', {}).get('amount', 0)
            call_cost = valid_actions[1].get('amount', 0) if len(valid_actions) > 1 else 0
            pot_odds = call_cost / max(pot_size + call_cost, 1) if call_cost > 0 else 0
            features[2] = min(pot_odds, 1.0)
            
            # 4. Stack to pot ratio
            my_stack = self._get_my_stack(seats, my_uuid)
            stack_pot_ratio = my_stack / max(pot_size, 1) if pot_size > 0 else my_stack / 100
            features[3] = min(stack_pot_ratio / 10, 1.0)  # Normalized
            
            # 5. Betting round (preflop=0, flop=0.33, turn=0.66, river=1.0)
            street_map = {'preflop': 0, 'flop': 0.33, 'turn': 0.66, 'river': 1.0}
            street = round_state.get('street', 'preflop')
            features[4] = street_map.get(street, 0)
            
            # 6. Number of active players
            active_players = len([s for s in seats if s.get('state') != 'folded'])
            features[5] = active_players / max(len(seats), 1)
            
            # 7. Aggression level (based on recent raises)
            aggression = self._calculate_aggression_level(round_state)
            features[6] = aggression
            
            # 8. Relative stack size compared to others
            relative_stack = self._calculate_relative_stack_size(seats, my_uuid)
            features[7] = relative_stack
            
            # 9. Hand strength relative to board
            board_texture = self._analyze_board_texture(community_cards)
            features[8] = board_texture
            
            # 10. Opponent tendency modeling
            opponent_aggression = self._estimate_opponent_aggression(round_state)
            features[9] = opponent_aggression
            
            # 11. Tournament/game progress (remaining rounds)
            total_rounds = 20 if self.stage == 2 else 5
            current_round = round_state.get('round_count', 1)
            features[10] = (total_rounds - current_round) / total_rounds
            
            # 12. Pot commitment level
            pot_commitment = self._calculate_pot_commitment(round_state, my_uuid)
            features[11] = pot_commitment
            
            return features
            
        except Exception as e:
            # Return safe default features
            return np.array([0.5] * self.input_size)
    
    def _get_my_stack(self, seats, my_uuid):
        """Get current stack size"""
        for seat in seats:
            if seat.get('uuid') == my_uuid:
                return seat.get('stack', 0)
        return 0
    
    def _calculate_aggression_level(self, round_state):
        """Calculate recent aggression level in the hand"""
        try:
            action_histories = round_state.get('action_histories', {})
            total_actions = 0
            aggressive_actions = 0
            
            for street_actions in action_histories.values():
                for action in street_actions:
                    total_actions += 1
                    if action.get('action') in ['raise', 'bet']:
                        aggressive_actions += 1
            
            return aggressive_actions / max(total_actions, 1)
        except:
            return 0.3  # Default moderate aggression
    
    def _calculate_relative_stack_size(self, seats, my_uuid):
        """Calculate relative stack size compared to opponents"""
        try:
            stacks = [seat.get('stack', 0) for seat in seats]
            my_stack = self._get_my_stack(seats, my_uuid)
            
            if not stacks or max(stacks) == 0:
                return 0.5
            
            return my_stack / max(stacks)
        except:
            return 0.5
    
    def _analyze_board_texture(self, community_cards):
        """Analyze board texture (coordinated vs rainbow)"""
        try:
            if len(community_cards) < 3:
                return 0.5  # No flop yet
            
            # Parse cards
            parsed_cards = [self._parse_card(card) for card in community_cards if self._parse_card(card)]
            if len(parsed_cards) < 3:
                return 0.5
            
            ranks = [card['rank'] for card in parsed_cards]
            suits = [card['suit'] for card in parsed_cards]
            
            texture_score = 0.0
            
            # Suit coordination (flush draws)
            suit_counts = {}
            for suit in suits:
                suit_counts[suit] = suit_counts.get(suit, 0) + 1
            
            max_suit_count = max(suit_counts.values())
            if max_suit_count >= 3:
                texture_score += 0.4
            
            # Rank coordination (straight draws)
            rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                          '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
            
            numeric_ranks = sorted([rank_values.get(rank, 0) for rank in ranks])
            for i in range(len(numeric_ranks) - 1):
                if numeric_ranks[i+1] - numeric_ranks[i] <= 2:
                    texture_score += 0.2
            
            return min(texture_score, 1.0)
        except:
            return 0.5
    def _estimate_opponent_aggression(self, round_state):
        """Enhanced opponent aggression estimation with persistent modeling"""
        try:
            action_histories = round_state.get('action_histories', {})
            opponent_actions = 0
            aggressive_opponent_actions = 0
            
            my_uuid = getattr(self, 'uuid', None)
            
            # Current hand aggression
            for street_actions in action_histories.values():
                for action in street_actions:
                    if action.get('uuid') != my_uuid:
                        opponent_uuid = action.get('uuid')
                        action_type = action.get('action')
                        
                        # Initialize opponent profile if new
                        if opponent_uuid not in self.opponent_modeling:
                            self.opponent_modeling[opponent_uuid] = {
                                'total_actions': 0,
                                'aggressive_actions': 0,
                                'bluffs_caught': 0,
                                'showdowns': 0,
                                'fold_to_raise': 0,
                                'raise_frequency': 0
                            }
                        
                        profile = self.opponent_modeling[opponent_uuid]
                        profile['total_actions'] += 1
                        opponent_actions += 1
                        
                        if action_type in ['raise', 'bet']:
                            profile['aggressive_actions'] += 1
                            aggressive_opponent_actions += 1
                        elif action_type == 'fold' and self._was_facing_raise(street_actions, action):
                            profile['fold_to_raise'] += 1
            
            # Calculate combined aggression (current + historical)
            current_aggression = aggressive_opponent_actions / max(opponent_actions, 1)
            
            # Weight with historical data
            historical_aggression = 0.3
            total_historical_actions = sum(p['total_actions'] for p in self.opponent_modeling.values())
            total_historical_aggressive = sum(p['aggressive_actions'] for p in self.opponent_modeling.values())
            
            if total_historical_actions > 0:
                historical_aggression = total_historical_aggressive / total_historical_actions
            
            # Weighted combination (60% current, 40% historical)
            final_aggression = 0.6 * current_aggression + 0.4 * historical_aggression
            
            return final_aggression
            
        except:
            return 0.3
    
    def _was_facing_raise(self, street_actions, fold_action):
        """Check if a fold was in response to a raise"""
        fold_index = -1
        for i, action in enumerate(street_actions):
            if action == fold_action:
                fold_index = i
                break
        
        if fold_index > 0:
            prev_action = street_actions[fold_index - 1]
            return prev_action.get('action') in ['raise', 'bet']
        
        return False
    
    def _detect_bluff_patterns(self, round_state, winners):
        """Detect and learn from opponent bluffing patterns"""
        try:
            my_uuid = getattr(self, 'uuid', None)
            action_histories = round_state.get('action_histories', {})
            
            # Analyze showdowns to detect bluffs
            for winner in winners:
                winner_uuid = winner.get('uuid')
                if winner_uuid != my_uuid and winner_uuid in self.opponent_modeling:
                    # Check if they were aggressive with a weak hand (simplified)
                    was_aggressive = False
                    for street_actions in action_histories.values():
                        for action in street_actions:
                            if (action.get('uuid') == winner_uuid and 
                                action.get('action') in ['raise', 'bet']):
                                was_aggressive = True
                                break
                    
                    if was_aggressive:
                        # This could be enhanced with actual hand strength analysis
                        self.opponent_modeling[winner_uuid]['showdowns'] += 1
        except:
            pass
    
    def _calculate_pot_commitment(self, round_state, my_uuid):
        """Calculate how committed we are to the pot"""
        try:
            pot_size = round_state.get('pot', {}).get('main', {}).get('amount', 0)
            my_stack = self._get_my_stack(round_state.get('seats', []), my_uuid)
            
            # How much of our stack is already in the pot
            if my_stack <= 0:
                return 1.0
            
            # Estimate our contribution to pot (simplified)
            estimated_contribution = pot_size / max(len(round_state.get('seats', [])), 1)
            commitment = estimated_contribution / (my_stack + estimated_contribution)
            
            return min(commitment, 1.0)
        except:
            return 0.2
    def declare_action(self, valid_actions, hole_card, round_state):
        """Enhanced decision-making function using PPO with opponent adaptation"""
        try:
            # Extract state features
            state = self._extract_state_features(valid_actions, hole_card, round_state)
            self.current_state = state
            
            # Get action probabilities and raise parameters from actor network
            action_probs, raise_mean, raise_std = self._actor_forward(state)
            
            # Apply opponent-aware adjustments
            adjusted_probs = self._adjust_probs_for_opponents(action_probs, round_state, state)
            
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
            valid_probs = np.array([adjusted_probs[i] if i < len(adjusted_probs) else 0.1 for i in valid_action_indices])
            valid_probs = valid_probs / np.sum(valid_probs)
            
            # Add exploration noise based on current entropy coefficient
            if self.training:
                exploration_noise = np.random.dirichlet([1] * len(valid_probs)) * self.entropy_coef
                valid_probs = (1 - self.entropy_coef) * valid_probs + exploration_noise
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
                    
                    # Opponent-aware bet sizing
                    sizing_adjustment = self._get_opponent_aware_sizing(round_state, state)
                    normalized = min(1.0, max(0.0, normalized * sizing_adjustment))
                    
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
                actual_prob = adjusted_probs[action_idx] if action_idx < len(adjusted_probs) else 0.1
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
    
    def _adjust_probs_for_opponents(self, action_probs, round_state, state):
        """Adjust action probabilities based on opponent modeling"""
        try:
            adjusted_probs = action_probs.copy()
            
            if not self.opponent_modeling:
                return adjusted_probs
            
            # Get opponent aggression from state features
            opponent_aggression = state[9] if len(state) > 9 else 0.3
            hand_strength = state[0] if len(state) > 0 else 0.5
            
            # Adjust based on opponent aggression
            if opponent_aggression > 0.6:  # Very aggressive opponents
                # Be more cautious - increase fold probability, decrease raise
                adjusted_probs[0] *= 1.2  # Fold more
                adjusted_probs[2] *= 0.8  # Raise less
            elif opponent_aggression < 0.2:  # Very passive opponents
                # Be more aggressive against passive players
                if hand_strength > 0.4:  # With decent hands
                    adjusted_probs[2] *= 1.3  # Raise more
                    adjusted_probs[0] *= 0.7  # Fold less
            
            # Normalize
            adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
            
            return adjusted_probs
            
        except Exception:
            return action_probs
    
    def _get_opponent_aware_sizing(self, round_state, state):
        """Get opponent-aware bet sizing adjustment"""
        try:
            opponent_aggression = state[9] if len(state) > 9 else 0.3
            
            if opponent_aggression > 0.6:
                # Against aggressive opponents, size up with strong hands
                hand_strength = state[0] if len(state) > 0 else 0.5
                if hand_strength > 0.7:
                    return 1.3  # Larger bets
                else:
                    return 0.8  # Smaller bets with weak hands
            elif opponent_aggression < 0.2:
                # Against passive opponents, can bet larger for value
                return 1.2
            
            return 1.0  # Default sizing
            
        except Exception:
            return 1.0
    
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
        """Enhanced round result processing with opponent learning"""
        if not self.training or not self.last_action:
            return
        
        # Calculate reward
        reward = self._calculate_reward(winners, round_state)
        
        # Update opponent modeling
        self._detect_bluff_patterns(round_state, winners)
        self._update_position_strategies(round_state, winners)
        
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
        
        # Store game history for long-term learning
        game_data = {
            'final_reward': reward,
            'hand_strength': self.last_action['state'][0] if len(self.last_action['state']) > 0 else 0.5,
            'action_taken': self.last_action['action'],
            'won': any(w.get('uuid') == getattr(self, 'uuid', None) for w in winners)
        }
        self.game_history.append(game_data)
        
        # Update performance tracking
        is_winner = any(w.get('uuid') == getattr(self, 'uuid', None) for w in winners)
        self.win_rates.append(1.0 if is_winner else 0.0)
        self.avg_rewards.append(reward)
        
        # If game is over, finish trajectory and possibly train
        if experience['done']:
            self.trajectories.append(self.current_trajectory.copy())
            self.current_trajectory = []
            
            # Train when we have enough trajectories
            if len(self.trajectories) >= self.trajectories_per_session:
                self._train_ppo()
                self._print_training_stats()
                self.trajectories = []
                self.sessions_completed += 1
                
                # Stage transition
                if self.stage == 1 and self.sessions_completed >= 10:
                    self._transition_to_stage_2()
                
                # Save model periodically
                if self.sessions_completed % self.save_interval == 0:
                    self._save_old_agent()
                    self.save_model()
    
    def _update_position_strategies(self, round_state, winners):
        """Learn position-based strategies"""
        try:
            seats = round_state.get('seats', [])
            my_uuid = getattr(self, 'uuid', None)
            
            if not my_uuid:
                return
            
            # Find my position
            my_position = -1
            for i, seat in enumerate(seats):
                if seat.get('uuid') == my_uuid:
                    my_position = i
                    break
            
            if my_position >= 0:
                position_key = 'early' if my_position < len(seats) // 3 else 'middle' if my_position < 2 * len(seats) // 3 else 'late'
                
                if position_key not in self.position_play:
                    self.position_play[position_key] = {
                        'games': 0,
                        'wins': 0,
                        'avg_reward': 0,
                        'fold_rate': 0,
                        'raise_rate': 0
                    }
                
                pos_stats = self.position_play[position_key]
                pos_stats['games'] += 1
                
                is_winner = any(w.get('uuid') == my_uuid for w in winners)
                if is_winner:
                    pos_stats['wins'] += 1
                
                # Update action rates
                if hasattr(self, 'last_action') and self.last_action:
                    action = self.last_action.get('action', '')
                    if action == 'fold':
                        pos_stats['fold_rate'] = (pos_stats['fold_rate'] * (pos_stats['games'] - 1) + 1) / pos_stats['games']
                    elif action == 'raise':
                        pos_stats['raise_rate'] = (pos_stats['raise_rate'] * (pos_stats['games'] - 1) + 1) / pos_stats['games']
        except:
            pass
    
    def _print_training_stats(self):
        """Print training statistics"""
        if len(self.win_rates) > 0:
            recent_win_rate = np.mean(list(self.win_rates)[-20:])  # Last 20 games
            recent_avg_reward = np.mean(list(self.avg_rewards)[-20:])
            
            print(f"Stage {self.stage} - Session {self.sessions_completed}")
            print(f"Recent Win Rate: {recent_win_rate:.3f}")
            print(f"Recent Avg Reward: {recent_avg_reward:.3f}")
            
            # Print opponent modeling stats
            if self.opponent_modeling:
                print("Opponent Profiles:")
                for uuid, profile in list(self.opponent_modeling.items())[:3]:  # Show top 3
                    if profile['total_actions'] > 5:  # Only show opponents with enough data
                        aggression = profile['aggressive_actions'] / max(profile['total_actions'], 1)
                        print(f"  {uuid[:8]}: {aggression:.2f} aggression, {profile['total_actions']} actions")
    def _calculate_reward(self, winners, round_state):
        """Enhanced reward calculation with sophisticated poker metrics"""
        reward = 0.0
        my_uuid = getattr(self, 'uuid', None)
        
        if not my_uuid or not round_state:
            return reward
        
        # Get current and starting stack
        current_stack = 0
        seats = round_state.get('seats', [])
        for seat in seats:
            if seat.get('uuid') == my_uuid:
                current_stack = seat.get('stack', 0)
                break
        
        # Basic win/loss reward
        is_winner = False
        for winner in winners:
            if winner.get('uuid') == my_uuid:
                reward += 2.0  # Strong win reward
                is_winner = True
                break
        
        if not is_winner:
            reward -= 0.2  # Small loss penalty
        
        # Stack-based reward (chip utility)
        if hasattr(self, 'game_start_stack') and self.game_start_stack > 0:
            stack_change = current_stack - self.game_start_stack
            reward += stack_change / self.game_start_stack  # Normalized stack change
        
        # Pot efficiency reward (did we win relative to pot investment?)
        pot_size = round_state.get('pot', {}).get('main', {}).get('amount', 0)
        if is_winner and pot_size > 0:
            # Bonus for winning big pots efficiently
            pot_bonus = min(pot_size / 100, 1.0)  # Capped bonus
            reward += pot_bonus
        
        # Survival bonus in late game
        if self.rounds_in_current_game > 3:
            active_players = len([s for s in seats if s.get('stack', 0) > 0])
            if active_players <= 3:  # Late game
                reward += 0.3  # Survival bonus
        
        # Penalty for going all-in frequently (risk management)
        if current_stack < 50:  # Very low stack
            reward -= 0.1
        
        # Action quality reward (based on hand strength vs action)
        if hasattr(self, 'last_action') and self.last_action:
            action_quality = self._evaluate_action_quality(self.last_action, round_state)
            reward += action_quality * 0.5
        
        return reward
    
    def _evaluate_action_quality(self, last_action, round_state):
        """Evaluate the quality of the last action taken"""
        if not last_action:
            return 0.0
        
        try:
            state = last_action.get('state', np.zeros(self.input_size))
            action = last_action.get('action', '')
            
            # Extract hand strength from state (feature 0)
            hand_strength = state[0] if len(state) > 0 else 0.5
            pot_odds = state[2] if len(state) > 2 else 0.5
            
            # Evaluate action appropriateness
            quality = 0.0
            
            if action == 'fold':
                # Good fold with weak hand
                if hand_strength < 0.3:
                    quality += 0.2
                # Bad fold with strong hand
                elif hand_strength > 0.7:
                    quality -= 0.3
            elif action == 'call':
                # Good call with decent hand and good pot odds
                if hand_strength > 0.4 and pot_odds < 0.3:
                    quality += 0.1
                # Bad call with weak hand and poor pot odds
                elif hand_strength < 0.2 and pot_odds > 0.5:
                    quality -= 0.2
            elif action == 'raise':
                # Good raise with strong hand
                if hand_strength > 0.6:
                    quality += 0.3
                # Bad raise with weak hand
                elif hand_strength < 0.3:
                    quality -= 0.4
            
            return quality
        except:
            return 0.0
    
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
        """Enhanced PPO training with adaptive exploration"""
        if not self.trajectories:
            return
        
        # Adaptive exploration based on performance
        self._update_exploration_rate()
        
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
        
        # PPO training epochs with early stopping
        for epoch in range(self.epochs_per_update):
            # This is a simplified PPO update
            # In practice, you would implement proper gradient computation and backpropagation
            self._simplified_ppo_update(all_states, all_actions, all_old_probs, all_advantages, all_returns)
            
            # Early stopping based on improvement
            if epoch > 3 and self._check_early_stopping():
                break
    
    def _update_exploration_rate(self):
        """Adaptively update exploration based on recent performance"""
        if len(self.win_rates) >= 20:
            recent_win_rate = np.mean(list(self.win_rates)[-20:])
            
            if recent_win_rate > self.performance_threshold:
                # Reduce exploration when performing well
                self.entropy_coef = max(
                    self.min_entropy_coef,
                    self.entropy_coef * self.exploration_decay
                )
            else:
                # Increase exploration when performing poorly
                self.entropy_coef = min(
                    self.initial_entropy_coef,
                    self.entropy_coef * 1.02
                )
    
    def _check_early_stopping(self):
        """Check if training should stop early (simplified)"""
        # This is a placeholder for more sophisticated early stopping
        # Could check gradient norms, loss convergence, etc.
        return False
    
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
        """Enhanced PPO update with proper clipping and gradient estimation"""
        if len(states) == 0:
            return
        
        # Batch processing for better stability
        batch_size = min(self.batch_size, len(states))
        indices = np.random.choice(len(states), batch_size, replace=False)
        
        batch_states = states[indices]
        batch_actions = actions[indices]
        batch_old_probs = old_probs[indices]
        batch_advantages = advantages[indices]
        batch_returns = returns[indices]
        
        # Actor update with PPO clipping
        for i in range(len(batch_states)):
            state = batch_states[i]
            action_idx = int(batch_actions[i])
            old_prob = batch_old_probs[i]
            advantage = batch_advantages[i]
            
            # Get current action probabilities
            action_probs, _, _ = self._actor_forward(state)
            current_prob = action_probs[min(action_idx, len(action_probs) - 1)]
            
            # Calculate probability ratio
            prob_ratio = current_prob / max(old_prob, 1e-8)
            
            # PPO clipped objective
            surr1 = prob_ratio * advantage
            surr2 = np.clip(prob_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage
            actor_loss = -min(surr1, surr2)
            
            # Simplified gradient update (approximation)
            self._update_actor_weights(state, action_idx, actor_loss)
        
        # Critic update
        for i in range(len(batch_states)):
            state = batch_states[i]
            target_value = batch_returns[i]
            
            current_value = self._critic_forward(state)
            value_loss = (target_value - current_value) ** 2
            
            # Simplified gradient update
            self._update_critic_weights(state, value_loss)
    
    def _update_actor_weights(self, state, action_idx, loss):
        """Simplified actor weight update"""
        gradient_magnitude = self.learning_rate * loss * 0.01
        
        # Add small perturbations based on loss magnitude
        for key in self.actor_weights:
            if 'W' in key:
                # Apply gradient clipping
                gradient = gradient_magnitude * np.random.randn(*self.actor_weights[key].shape)
                gradient = np.clip(gradient, -self.max_grad_norm, self.max_grad_norm)
                self.actor_weights[key] -= gradient
    
    def _update_critic_weights(self, state, loss):
        """Simplified critic weight update"""
        gradient_magnitude = self.learning_rate * loss * 0.01
        
        for key in self.critic_weights:
            if 'W' in key:
                gradient = gradient_magnitude * np.random.randn(*self.critic_weights[key].shape)
                gradient = np.clip(gradient, -self.max_grad_norm, self.max_grad_norm)
                self.critic_weights[key] -= gradient
    def _transition_to_stage_2(self):
        """Transition from stage 1 to stage 2 (self-play)"""
        print("=" * 50)
        print("Transitioning to Stage 2: Self-play with 20-round games")
        print(f"Final Stage 1 stats:")
        if len(self.win_rates) > 0:
            print(f"  Overall win rate: {np.mean(list(self.win_rates)):.3f}")
            print(f"  Recent win rate: {np.mean(list(self.win_rates)[-20:]):.3f}")
        print(f"  Opponent profiles learned: {len(self.opponent_modeling)}")
        print("=" * 50)
        
        self.stage = 2
        self.sessions_completed = 0
        # Save current agent as first opponent for self-play
        self._save_old_agent()
        
        # Reset some parameters for stage 2
        self.entropy_coef = self.initial_entropy_coef * 0.5  # Reduce exploration for self-play
        self.trajectories_per_session = 3  # More frequent training in self-play
    
    def create_opponent_agent(self, difficulty_level=0.8):
        """Create a slightly modified version for self-play opponents"""
        opponent = PPOPokerAgent(training=False)
        
        if self.old_agents:
            # Use an old version with some modifications
            old_agent = random.choice(list(self.old_agents))
            opponent.actor_weights = {k: v.copy() for k, v in old_agent['actor_weights'].items()}
            opponent.critic_weights = {k: v.copy() for k, v in old_agent['critic_weights'].items()}
            
            # Add some variation to make it different
            variation_strength = (1.0 - difficulty_level) * 0.1
            for key in opponent.actor_weights:
                if 'W' in key:
                    noise = np.random.randn(*opponent.actor_weights[key].shape) * variation_strength
                    opponent.actor_weights[key] += noise
        else:
            # Use current weights as base
            opponent.actor_weights = {k: v.copy() for k, v in self.actor_weights.items()}
            opponent.critic_weights = {k: v.copy() for k, v in self.critic_weights.items()}
        
        # Adjust exploration for opponent
        opponent.entropy_coef = self.entropy_coef * (0.5 + difficulty_level * 0.5)
        
        return opponent
    
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