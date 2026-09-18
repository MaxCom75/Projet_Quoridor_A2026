from player_quoridor import PlayerQuoridor
from game_state_quoridor import GameStateQuoridor
from typing import Optional
from seahorse.utils.custom_exceptions import MethodNotImplementedError
from seahorse.game.action import Action
from seahorse.player.player import Player
from queue import PriorityQueue

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
        actions = tuple(current_state.generate_possible_stateless_actions())
        
        if not actions:
            raise RuntimeError("No legal action available.")

        best_action = None
        best_cost = 100

        prefer_placing_wall = False

        my_player = 0
        ennemy = 1
        if current_state.active_player.id == current_state.players[1].id:
            my_player = 1
            ennemy = 0

        my_shortest_path = current_state._shortest_path(current_state.players[my_player])
        ennemy_shortest_path = current_state._shortest_path(current_state.players[ennemy])
        if my_shortest_path > ennemy_shortest_path:
            prefer_placing_wall = True
            best_cost = 0

        for action in actions:
            temp_state = current_state.apply_action(action)
            if prefer_placing_wall:
                cost = temp_state._shortest_path(temp_state.players[ennemy])
                if cost > best_cost:
                    best_cost = cost
                    best_action = action
            else:
                cost = temp_state._shortest_path(temp_state.players[my_player])
                if cost < best_cost:
                    best_cost = cost
                    best_action = action

        return best_action