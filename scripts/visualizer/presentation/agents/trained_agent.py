"""
Trained Agent Wrapper

Wraps trained DQN/PPO models for use in presentation recording.
Provides unified interface matching baseline agents.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
import glob

import torch
import torch.nn as nn
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from core.networks import DQN_MLP, DuelingDQN_MLP, NoisyDQN_MLP, PPO_Actor_MLP


# Feature dimension mapping
FEATURE_DIMS = {
    'basic': 11,
    'flood-fill': 14,
    'floodfill': 14,
    'selective': 19,
    'enhanced': 24
}


class TrainedAgent:
    """
    Wrapper for trained RL models.

    Provides get_action() interface matching baseline agents,
    plus Q-value access for visualization.
    """

    def __init__(
        self,
        weights_path: str,
        network_type: str = 'dqn',
        feature_mode: str = 'basic',
        hidden_dims: Tuple[int, ...] = (128, 128),
        device: str = 'cuda'
    ):
        """
        Initialize trained agent.

        Args:
            weights_path: Path to model weights (supports glob patterns)
            network_type: 'dqn', 'double_dqn', 'dueling', 'noisy', 'per', 'ppo'
            feature_mode: 'basic', 'flood-fill', 'selective', 'enhanced'
            hidden_dims: Hidden layer dimensions
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.feature_mode = feature_mode
        self.network_type = network_type

        # Get input dimension from feature mode
        self.input_dim = FEATURE_DIMS.get(feature_mode, 11)

        # Resolve glob pattern if needed
        if '*' in weights_path:
            matches = glob.glob(weights_path)
            if not matches:
                raise FileNotFoundError(f"No weights found matching: {weights_path}")
            weights_path = sorted(matches)[-1]  # Get most recent
            print(f"Using weights: {weights_path}")

        self.weights_path = weights_path

        # Create network
        self.network = self._create_network(network_type, hidden_dims)

        # Load weights
        self._load_weights(weights_path)

        # Set to eval mode
        self.network.to(self.device)
        self.network.eval()

        # Store last Q-values for visualization
        self._last_q_values: Optional[np.ndarray] = None
        self._last_action: Optional[int] = None

    def _create_network(
        self,
        network_type: str,
        hidden_dims: Tuple[int, ...]
    ) -> nn.Module:
        """Create network architecture based on type."""
        output_dim = 3  # relative actions: straight, left, right

        if network_type in ('dueling', 'dueling_dqn'):
            return DuelingDQN_MLP(
                input_dim=self.input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims
            )
        elif network_type in ('noisy', 'noisy_dqn'):
            return NoisyDQN_MLP(
                input_dim=self.input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims
            )
        elif network_type == 'ppo':
            return PPO_Actor_MLP(
                input_dim=self.input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims
            )
        else:
            # Default DQN (also for double_dqn, per_dqn - same architecture)
            return DQN_MLP(
                input_dim=self.input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims
            )

    def _load_weights(self, weights_path: str):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(weights_path, map_location=self.device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'policy_net' in checkpoint:
                state_dict = checkpoint['policy_net']
            elif 'actor' in checkpoint:
                state_dict = checkpoint['actor']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # Assume checkpoint is the state dict itself
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        self.network.load_state_dict(state_dict)

    def get_action(self, observation: np.ndarray) -> int:
        """
        Get action from trained model.

        Args:
            observation: Feature observation array

        Returns:
            Action index (0=straight, 1=left, 2=right)
        """
        with torch.no_grad():
            if isinstance(observation, np.ndarray):
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            else:
                obs_tensor = observation.unsqueeze(0) if observation.dim() == 1 else observation
                obs_tensor = obs_tensor.to(self.device)

            if self.network_type == 'ppo':
                # PPO outputs logits, take argmax
                logits = self.network(obs_tensor)
                action = logits.argmax(dim=1).item()
                # Store probabilities as "Q-values" for visualization
                probs = torch.softmax(logits, dim=1)
                self._last_q_values = probs[0].cpu().numpy()
            else:
                # DQN variants output Q-values
                q_values = self.network(obs_tensor)
                action = q_values.argmax(dim=1).item()
                self._last_q_values = q_values[0].cpu().numpy()

            self._last_action = action
            return action

    def get_q_values(self, observation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get Q-values for current or given observation.

        Args:
            observation: Optional observation (uses last if None)

        Returns:
            Q-values array [straight, left, right]
        """
        if observation is not None:
            # Compute Q-values for given observation
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

                if self.network_type == 'ppo':
                    logits = self.network(obs_tensor)
                    return torch.softmax(logits, dim=1)[0].cpu().numpy()
                else:
                    q_values = self.network(obs_tensor)
                    return q_values[0].cpu().numpy()

        # Return cached Q-values from last action
        if self._last_q_values is None:
            return np.zeros(3)
        return self._last_q_values

    @property
    def last_action(self) -> Optional[int]:
        """Get last selected action."""
        return self._last_action

    @property
    def action_names(self) -> list:
        """Get action names."""
        return ['STRAIGHT', 'LEFT', 'RIGHT']

    def __repr__(self) -> str:
        return (f"TrainedAgent(network={self.network_type}, "
                f"features={self.feature_mode}, input_dim={self.input_dim})")


def load_agent_from_path(weights_path: str) -> TrainedAgent:
    """
    Auto-detect agent configuration from weights filename.

    Parses filename to determine:
    - Network type (dqn, double_dqn, dueling, noisy, per, ppo)
    - Feature mode (basic, flood-fill, selective, enhanced)
    - Hidden dimensions

    Args:
        weights_path: Path to weights file

    Returns:
        Configured TrainedAgent
    """
    filename = Path(weights_path).stem.lower()

    # Detect network type
    if 'dueling' in filename:
        network_type = 'dueling'
    elif 'noisy' in filename:
        network_type = 'noisy'
    elif 'double' in filename:
        network_type = 'double_dqn'
    elif 'per' in filename:
        network_type = 'per_dqn'
    elif 'ppo' in filename:
        network_type = 'ppo'
    elif 'a2c' in filename:
        network_type = 'ppo'  # Same actor architecture
    else:
        network_type = 'dqn'

    # Detect feature mode
    if 'enhanced' in filename:
        feature_mode = 'enhanced'
    elif 'selective' in filename:
        feature_mode = 'selective'
    elif 'flood' in filename or 'floodfill' in filename:
        feature_mode = 'flood-fill'
    else:
        feature_mode = 'basic'

    # Detect hidden dimensions
    if '256x256' in filename or 'large' in filename:
        hidden_dims = (256, 256)
    else:
        hidden_dims = (128, 128)

    return TrainedAgent(
        weights_path=weights_path,
        network_type=network_type,
        feature_mode=feature_mode,
        hidden_dims=hidden_dims
    )
