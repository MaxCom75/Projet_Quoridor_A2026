from player_quoridor import PlayerQuoridor
from game_state_quoridor import GameStateQuoridor
from typing import Optional, List
from seahorse.utils.custom_exceptions import MethodNotImplementedError
from seahorse.game.action import Action
from seahorse.player.player import Player
from queue import PriorityQueue
import math
import time
import random

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
        def max_value(state: GameStateQuoridor, state_score, alpha, beta, depth):
            if time.time() > deadline:
                raise SearchTimeout()
            if state.is_done():
                return terminal_value(state, depth), None
            neighbors = generateNeighbors(state, state_score, maximize=True) if depth > 0 else []
            if not neighbors:
                return heuristic(state), None
            v = -math.inf
            m = None
            for action, s1, s1_score in neighbors:
                (v1, _) = min_value(s1, s1_score, alpha, beta, depth - 1)
                if v1 > v:
                    v = v1
                    m = action
                    alpha = max(alpha, v)
                if v >= beta:
                    return (v, m)
            return (v, m)

        def min_value(state: GameStateQuoridor, state_score, alpha, beta, depth):
            if time.time() > deadline:
                raise SearchTimeout()
            if state.is_done():
                return terminal_value(state, depth), None
            neighbors = generateNeighbors(state, state_score, maximize=False) if depth > 0 else []
            if not neighbors:
                return heuristic(state), None
            v = math.inf
            m = None
            for action, s1, s1_score in neighbors:
                (v1, _) = max_value(s1, s1_score, alpha, beta, depth - 1)
                if v1 < v:
                    v = v1
                    m = action
                    beta = min(beta, v)
                if v <= alpha:
                    return (v, m)
            return (v, m)

        def generateNeighbors(state: GameStateQuoridor, state_score, maximize: bool):
            PAWN_RADIUS = 1
            WALL_RADIUS = 2
            pawns = list(state.rep.pawn_positions.values())
            walls = [(w.row, w.col) for w in state.rep.walls]

            def is_near(pos, others, radius):
                return any(max(abs(pos[0] - o[0]), abs(pos[1] - o[1])) <= radius for o in others)

            neighbors = []
            for action in state.generate_possible_stateless_actions():
                destination = action.data["destination"]
                if action.data["type"] == "move":
                    chosen = True
                else:
                    chosen = is_near(destination, pawns, PAWN_RADIUS) or is_near(destination, walls, WALL_RADIUS)
                if chosen:
                    s1 = state.apply_action(action)
                    s1_score = heuristic(s1)
                    if (maximize and s1_score >= state_score) or (not maximize and s1_score <= state_score):
                        neighbors.append((action, s1, s1_score))
                    else:
                        delta = (state_score - s1_score) if maximize else (s1_score - state_score)
                        if random.random() < math.exp(-delta / t):
                            neighbors.append((action, s1, s1_score))
            return neighbors

        def terminal_value(state: GameStateQuoridor, depth):
            return heuristic(state) * (1 + depth)

        def heuristic(state: GameStateQuoridor):
            if state.is_done():
                return 1 if state.get_player_score(self) == 1.0 else -1
            opponent = state._opponent(self)
            my_dist = state._shortest_path(self)
            opp_dist = state._shortest_path(opponent)
            WALL_VALUE = 0.5
            MAX_WALLS = 10
            wall_diff = state.rep.remaining_walls[self.id] - state.rep.remaining_walls[opponent.id]

            my_turn = state.get_active_player().get_id() == self.id
            ahead = my_dist < opp_dist or (my_dist == opp_dist and my_turn)

            if ahead:
                max_dist = state.rep.dimension ** 2
                tie_range = max_dist + 2 * WALL_VALUE * MAX_WALLS + 1
                tie_break = (opp_dist + WALL_VALUE * (wall_diff + MAX_WALLS)) / tie_range
                key = -my_dist + tie_break
                return (key + max_dist + 1) / (max_dist + 3)

            return (math.tanh((opp_dist - my_dist + WALL_VALUE * wall_diff) / 5) - 1) / 2
          
        MAX_DEPTH = 20
        EFFECTIVE_BRANCHING = 3
        MIN_RESTARTS = 3
        SAFETY = 0.8

        TOTAL_TIME = 15 * 60
        T_START = 0.1
        T_END = 0.001
        progress = 1 - remaining_time / TOTAL_TIME
        t = T_START * (T_END / T_START) ** progress

        moves_left = current_state._shortest_path(self) + current_state.rep.remaining_walls[self.id]
        budget = remaining_time / (moves_left + 5)
        start = time.time()
        deadline = start + budget

        best_action = next(iter(current_state.generate_possible_stateless_actions()))

        try:
            _, action = max_value(current_state, heuristic(current_state), -math.inf, math.inf, 1)
            if action is not None:
                best_action = action
        except SearchTimeout:
            return best_action
        time_depth1 = max(time.time() - start, 1e-4)

        usable = SAFETY * (deadline - time.time())
        depth = 1
        while depth < MAX_DEPTH and time_depth1 * EFFECTIVE_BRANCHING ** depth * MIN_RESTARTS <= usable:
            depth += 1
        print(f"############DEPTH::{depth}#############")
            
        best_action_value = -math.inf
        i = 1
        while True:
            try:
                print(f"######## RESTART : {i} ########")
                start = time.time()
                value, action = max_value(current_state, heuristic(current_state), -math.inf, math.inf, depth)
                end = time.time()
                duration = end - start
                i += 1
            except SearchTimeout:
                break
            if action is None:
                continue
            if value > best_action_value:
                best_action_value = value
                best_action = action
            if end + duration > deadline:
                break
        return best_action
