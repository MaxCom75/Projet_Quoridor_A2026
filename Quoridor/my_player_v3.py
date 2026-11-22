from player_quoridor import PlayerQuoridor
from game_state_quoridor import GameStateQuoridor
from typing import Optional, List
from seahorse.utils.custom_exceptions import MethodNotImplementedError
from seahorse.game.action import Action
from seahorse.player.player import Player
from queue import PriorityQueue
import math
import time

class SearchTimeout(Exception):
    pass

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
            if time.time() > deadline:
                raise SearchTimeout()
            if state.is_done():
                return terminal_value(state, depth), None
            neighbors = generateNeighbors(state, maximize=True) if depth > 0 else []
            if not neighbors:
                return heuristic(state), None
            v = -math.inf
            m = None
            for action, s1 in neighbors:
                (v1, _) = min_value(s1, alpha, beta, depth - 1)
                if v1 > v:
                    v = v1
                    m = action
                    alpha = max(alpha, v)
                if v >= beta:
                    return (v, m)
            return (v, m)

        def min_value(state: GameStateQuoridor, alpha, beta, depth):
            if time.time() > deadline:
                raise SearchTimeout()
            if state.is_done():
                return terminal_value(state, depth), None
            neighbors = generateNeighbors(state, maximize=False) if depth > 0 else []
            if not neighbors:
                return heuristic(state), None
            v = math.inf
            m = None
            for action, s1 in neighbors:
                (v1, _) = max_value(s1, alpha, beta, depth - 1)
                if v1 < v:
                    v = v1
                    m = action
                    beta = min(beta, v)
                if v <= alpha:
                    return (v, m)
            return (v, m)

        def generateNeighbors(state: GameStateQuoridor, maximize: bool):
            PAWN_RADIUS = 1
            WALL_RADIUS = 2
            pawns = list(state.rep.pawn_positions.values())
            walls = [(w.row, w.col) for w in state.rep.walls]

            def is_near(pos, others, radius):
                return any(max(abs(pos[0] - o[0]), abs(pos[1] - o[1])) <= radius for o in others)

            neighbors = []
            state_score = heuristic(state)
            for action in state.generate_possible_stateless_actions():
                destination = action.data["destination"]
                if action.data["type"] == "move":
                    chosen = True
                else:
                    chosen = is_near(destination, pawns, PAWN_RADIUS) or is_near(destination, walls, WALL_RADIUS)
                if chosen:
                    s1 = state.apply_action(action)
                    s1_score = heuristic(s1)
                    if (maximize and s1_score > state_score) or (not maximize and s1_score < state_score):
                        neighbors.append((action, s1))
            return neighbors

        def terminal_value(state: GameStateQuoridor, depth):
            # Scale by remaining depth so a win found sooner (or a loss found later) is preferred
            return heuristic(state) * (1 + depth)

        def heuristic(state: GameStateQuoridor):
            if state.is_done():
                return 1 if state.get_player_score(self) == 1.0 else -1
            opponent = state._opponent(self)
            my_dist = state._shortest_path(self)
            opp_dist = state._shortest_path(opponent)
            # A wall in hand is a threat to lengthen the opponent's path later, so it has value
            # Below 1 so a wall that only adds +1 still counts as an improvement for the neighbor filter
            WALL_VALUE = 0.5
            MAX_WALLS = 10
            wall_diff = state.rep.remaining_walls[self.id] - state.rep.remaining_walls[opponent.id]

            # Equal distances with me to move means I arrive first
            my_turn = state.get_active_player().get_id() == self.id
            ahead = my_dist < opp_dist or (my_dist == opp_dist and my_turn)

            if ahead:
                # Shortening my own path always beats anything else; opponent's path and walls are tie-breakers
                max_dist = state.rep.dimension ** 2
                tie_range = max_dist + 2 * WALL_VALUE * MAX_WALLS + 1
                tie_break = (opp_dist + WALL_VALUE * (wall_diff + MAX_WALLS)) / tie_range  # in [0, 1)
                key = -my_dist + tie_break  # in [-max_dist, 1)
                return (key + max_dist + 1) / (max_dist + 3)  # in (0, 1)

            # Not ahead: race difference plus walls in hand, mapped into (-1, 0) to stay below any ahead score
            return (math.tanh((opp_dist - my_dist + WALL_VALUE * wall_diff) / 5) - 1) / 2

        MAX_DEPTH = 20
        moves_left = current_state._shortest_path(self) + current_state.rep.remaining_walls[self.id]
        budget = remaining_time / (moves_left + 5)
        deadline = time.time() + budget

        best_action = next(iter(current_state.generate_possible_stateless_actions()))

        for depth in range(1, MAX_DEPTH + 1):
            try:
                action = max_value(current_state, -math.inf, math.inf, depth)[1]
            except SearchTimeout:
                break
            if action is not None:
                best_action = action
        return best_action