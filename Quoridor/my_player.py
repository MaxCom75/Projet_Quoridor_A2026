from player_quoridor import PlayerQuoridor
from game_state_quoridor import GameStateQuoridor
from typing import Optional, List
from seahorse.utils.custom_exceptions import MethodNotImplementedError
from seahorse.game.action import Action
from seahorse.player.player import Player
from queue import PriorityQueue
import math

class MyPlayer(PlayerQuoridor):
    """
    Player class for Quoridor game

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, goal_row: int=0, name: str = "bob", *args, **kwargs) -> None:
        """
        Initialize the PlayerQuoridor instance.

        Args:
            piece_type (str): Type of the player's game piece
            goal_row (int): The row the player wants to reach
            name (str, optional): Name of the player (default is "bob")
        """
        super().__init__(piece_type, goal_row, name)

    def compute_action(self, current_state: GameStateQuoridor, remaining_time: float = 15*60, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameStateQuoridor): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """
        def max_value(state: GameStateQuoridor, alpha, beta, depth):
            if state.is_done() or depth == 0:
                return heuristic(state), None
            v = -math.inf
            m = None
            for action, s1 in best_quarter(state, maximize=True):
                (v1, _) = min_value(s1, alpha, beta, depth - 1)
                if v1 > v:
                    v = v1
                    m = action
                    alpha = max(alpha, v)
                if v >= beta:
                    return (v, m)
            return (v, m)

        def min_value(state: GameStateQuoridor, alpha, beta, depth):
            if state.is_done() or depth == 0:
                return heuristic(state), None
            v = math.inf
            m = None
            for action, s1 in best_quarter(state, maximize=False):
                (v1, _) = max_value(s1, alpha, beta, depth - 1)
                if v1 < v:
                    v = v1
                    m = action
                    beta = min(beta, v)
                if v <= alpha:
                    return (v, m)
            return (v, m)

        def best_quarter(state: GameStateQuoridor, maximize: bool):
            children = []
            for action in state.get_possible_stateful_actions():
                s1 = action.get_next_game_state()
                children.append((heuristic(s1), action, s1))
            children.sort(key=lambda c: c[0], reverse=maximize)
            quarter = (len(children) + 3) // 4
            return [(action, s1) for _, action, s1 in children[:quarter]]

        def heuristic(state: GameStateQuoridor):
            if state.is_done():
                return 1 if state.get_player_score(self) == 1.0 else -1

            opponent = state._opponent(self)
            my_dist = state._shortest_path(self)
            opp_dist = state._shortest_path(opponent)

            if my_dist is None or opp_dist is None:
                return 0

            WALL_VALUE = 1.5
            walls = state.get_rep().remaining_walls
            wall_diff = walls[self.id] - walls[opponent.id]

            raw = (opp_dist - my_dist) + WALL_VALUE * wall_diff
            return math.tanh(raw / 5)

        return max_value(current_state, -math.inf, math.inf, 3)[1]