"""
An AI player for Othello. 
"""

from othello_shared import find_lines, get_possible_moves, get_score, play_move
import random
import sys
import time

# Caching
cache = {}

# You can use the functions in othello_shared to write your AI


def eprint(*args, **kwargs):  # you can use this for debugging, as it will print to sterr and not stdout
    print(*args, file=sys.stderr, **kwargs)

# Method to compute utility value of terminal state


def compute_utility(board, color):
    # IMPLEMENT
    score = get_score(board)
    if color == 1:
        return score[0] - score[1]
    return score[1] - score[0]

# Better heuristic value of board


def compute_heuristic(board, color):  # not implemented, optional
    # IMPLEMENT
    weight = 0
    opp_color = 1
    if color == 1:
        opp_color = 2

    sidelen = len(board)

    # Try to restrain opponent moves - mobility
    my_moves = get_possible_moves(board, color)
    opp_moves = get_possible_moves(board, opp_color)
    if len(my_moves) + len(opp_moves) > 0:
        weight += len(my_moves)-len(opp_moves)

    # Parity
    score = get_score(board)
    weight += 100 * (compute_utility(board, color) / (score[0] + score[1]))

    # Corners
    my_corners = 0
    opp_corners = 0
    nw_corner = board[0][0]
    ne_corner = board[0][sidelen-1]
    sw_corner = board[sidelen-1][0]
    se_corner = board[sidelen-1][sidelen-1]
    if nw_corner == color:
        my_corners += 1
    elif nw_corner == opp_color:
        opp_corners += 1

    if ne_corner == color:
        my_corners += 1
    elif ne_corner == opp_color:
        opp_corners += 1

    if sw_corner == color:
        my_corners += 1
    elif sw_corner == opp_color:
        opp_corners += 1

    if se_corner == color:
        my_corners += 1
    elif se_corner == opp_color:
        opp_corners += 1

    if my_corners + opp_corners > 0:
        weight += (my_corners-opp_corners)*10

    # If corners haven't been taken
    if nw_corner != color and nw_corner != opp_color:
        if board[1][1] == color:
            weight -= 5
        if board[1][1] == opp_color:
            weight += 5
    if ne_corner != color and ne_corner != opp_color:
        if board[1][sidelen-2] == color:
            weight -= 5
        if board[1][sidelen-2] == opp_color:
            weight += 5
    if sw_corner != color and sw_corner != opp_color:
        if board[sidelen-2][1] == color:
            weight -= 5
        if board[sidelen-2][1] == opp_color:
            weight += 5
    if se_corner != color and se_corner != opp_color:
        if board[sidelen-2][sidelen-2] == color:
            weight -= 5
        if board[sidelen-2][sidelen-2] == opp_color:
            weight += 5

    # Stability
    weight += stability(board, color, nw_corner, ne_corner,
                        sw_corner, se_corner, sidelen)
    weight -= stability(board, opp_color, nw_corner,
                        ne_corner, sw_corner, se_corner, sidelen)

    return weight

# Helper to check how many discs are stable based off the corner values


def stability(board, color, nw_corner, ne_corner, sw_corner, se_corner, sidelen):
    total = 0
    sboard = []
    for i in range(sidelen):
        row = []
        for j in range(sidelen):
            row.append(0)
        sboard.append(row)
    if nw_corner == color:
        sboard[0][0] = 1
    if ne_corner == color:
        sboard[0][sidelen-1] = 1
    if sw_corner == color:
        sboard[sidelen-1][0] = 1
    if se_corner == color:
        sboard[sidelen-1][sidelen-1] = 1

    for i in range(1, sidelen):
        for j in range(1, sidelen):
            if board[i-1][j] == color and sboard[i-1][j] == 1:
                sboard[i][j] = 1
            elif board[i-1][j-1] == color and sboard[i-1][j-1] == 1:
                sboard[i][j] = 1
            elif board[i][j-1] == color and sboard[i][j-1] == 1:
                sboard[i][j] = 1

    for i in range(sidelen):
        total += sum(sboard[i])
    return total

############ MINIMAX ###############################


def minimax_min_node(board, color, limit, caching=0):
    # IMPLEMENT (and replace the line below)
    opp_color = 1
    if color == 1:
        opp_color = 2

    # Change board into tuple so it can be used as a key in the cache dict
    if boardhash(board) in cache and caching == 1:
        return cache[boardhash(board)]

    moves = get_possible_moves(board, opp_color)
    # Depth base case
    if limit == 0:
        if caching == 1:
            cache[boardhash(board)] = ((-2, -2), compute_utility(board, color))
        return ((-2, -2), compute_utility(board, color))
    # No moves base case
    if len(moves) == 0:
        if caching == 1:
            cache[boardhash(board)] = ((-1, -1), compute_utility(board, color))
        return ((-1, -1), compute_utility(board, color))
    else:
        best = float('inf')
        best_move = None
        # Loop through possible moves to get best utility value
        for move in moves:
            newboard = play_move(board, opp_color, move[0], move[1])
            max_info = minimax_max_node(newboard, color, limit-1, caching)
            val = max_info[1]
            if val < best:
                best = val
                best_move = move
        if caching == 1:
            ccache[boardhash(board)] = (best_move, best)
    return (best_move, best)


def minimax_max_node(board, color, limit, caching=0):  # returns highest possible utility
    # IMPLEMENT (and replace the line below)

    # Change board into tuple so it can be used as a key in the cache dict
    if boardhash(board) in cache and caching == 1:
        return cache[boardhash(board)]

    moves = get_possible_moves(board, color)
    # Depth base case
    if limit == 0:
        if caching == 1:
            cache[boardhash(board)] = ((-2, -2), compute_utility(board, color))
        return ((-2, -2), compute_utility(board, color))
    # No moves base case
    if len(moves) == 0:
        if caching == 1:
            cache[boardhash(board)] = ((-1, -1), compute_utility(board, color))
        return ((-1, -1), compute_utility(board, color))
    else:
        best = float('-inf')
        best_move = None
        # Loop through possible moves to get best utility value
        for move in moves:
            newboard = play_move(board, color, move[0], move[1])
            min_info = minimax_min_node(newboard, color, limit-1, caching)
            val = min_info[1]
            if best < val:
                best = val
                best_move = move
        if caching == 1:
            cache[boardhash(board)] = (best_move, best)
    return (best_move, best)


def select_move_minimax(board, color, limit, caching=0):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.    
    """
    # IMPLEMENT (and replace the line below)
    move = minimax_max_node(board, color, limit, caching)
    return move[0]

############ ALPHA-BETA PRUNING #####################


def alphabeta_min_node(board, color, alpha, beta, limit, caching=0, ordering=0):
    opp_color = 1
    if color == 1:
        opp_color = 2

    if boardhash(board) in cache and caching == 1:
        return cache[boardhash(board)]

    # Get possible moves and order them based on best utility
    ordered_moves = []
    moves = get_possible_moves(board, opp_color)
    for move in moves:
        next_move = play_move(board, opp_color, move[0], move[1])
        util = compute_utility(next_move, color)
        ordered_moves.append((move, util, next_move))
    ordered_moves.sort(key=lambda x: x[1])

    # Depth base case
    if limit == 0:
        if caching == 1:
            cache[boardhash(board)] = ((), compute_utility(board, color))
        return ((), compute_utility(board, color))
    # No moves base case
    if len(moves) == 0:
        if caching == 1:
            cache[boardhash(board)] = ((), compute_utility(board, color))
        return ((), compute_utility(board, color))
    else:
        best = float('inf')
        best_move = None
        # Loop through possible moves by best utility value
        for item in ordered_moves:
            move = item[0]

            newboard = item[2]
            max_info = alphabeta_max_node(
                newboard, color, alpha, beta, limit-1, caching, ordering)
            val = max_info[1]
            # Update beta value
            if beta > val:
                beta = val

            if val < best:
                best = val
                best_move = move
            if caching == 1:
                cache[boardhash(board)] = (best_move, best)
            # Stop checking once cutoff is reached
            if alpha >= beta:
                break

    return (best_move, best)


def alphabeta_max_node(board, color, alpha, beta, limit, caching=0, ordering=0):
    # IMPLEMENT (and replace the line below)

    if boardhash(board) in cache and caching == 1:
        return cache[boardhash(board)]

    # Get possible moves and order them based on best utility
    ordered_moves = []
    moves = get_possible_moves(board, color)
    for move in moves:
        next_move = play_move(board, color, move[0], move[1])
        util = compute_utility(next_move, color)
        ordered_moves.append((move, util, next_move))
    ordered_moves.sort(key=lambda x: x[1], reverse=False)

    # Depth base case
    if limit == 0:
        if caching == 1:
            cache[boardhash(board)] = ((), compute_utility(board, color))
        return ((), compute_utility(board, color))
    # No moves base case
    if len(moves) == 0:
        if caching == 1:
            cache[boardhash(board)] = ((), compute_utility(board, color))
        return ((), compute_utility(board, color))
    else:
        best = float('-inf')
        best_move = None
        # Loop through possible moves by best utility value
        for item in ordered_moves:
            move = item[0]

            newboard = item[2]
            min_info = alphabeta_min_node(
                newboard, color, alpha, beta, limit-1, caching, ordering)
            val = min_info[1]

            # Update alpha value
            if alpha < val:
                alpha = val

            if best < val:
                best = val
                best_move = move
            if caching == 1:
                cache[boardhash(board)] = (best_move, best)
            # Stop checking once cutoff is reached
            if alpha >= beta:
                break
    return (best_move, best)

# Turn the board into nested tuples so they can be hashed


def boardhash(board):
    tboard = []
    for row in board:
        tboard.append(tuple(row))
    return tuple(tboard)


def select_move_alphabeta(board, color, limit, caching=0, ordering=0):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.    
    If ordering is ON (i.e. 1), use node ordering to expedite pruning and reduce the number of state evaluations. 
    If ordering is OFF (i.e. 0), do NOT use node ordering to expedite pruning and reduce the number of state evaluations. 
    """
    move = alphabeta_max_node(
        board, color, float("-Inf"), float("Inf"), limit, caching, ordering)
    return move[0]

####################################################


def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Othello AI")  # First line is the name of this AI
    arguments = input().split(",")

    # Player color: 1 for dark (goes first), 2 for light.
    color = int(arguments[0])
    limit = int(arguments[1])  # Depth limit
    minimax = int(arguments[2])  # Minimax or alpha beta
    caching = int(arguments[3])  # Caching
    ordering = int(arguments[4])  # Node-ordering (for alpha-beta only)

    if (minimax == 1):
        eprint("Running MINIMAX")
    else:
        eprint("Running ALPHA-BETA")

    if (caching == 1):
        eprint("State Caching is ON")
    else:
        eprint("State Caching is OFF")

    if (ordering == 1):
        eprint("Node Ordering is ON")
    else:
        eprint("Node Ordering is OFF")

    if (limit == -1):
        eprint("Depth Limit is OFF")
    else:
        eprint("Depth Limit is ", limit)

    if (minimax == 1 and ordering == 1):
        eprint("Node Ordering should have no impact on Minimax")

    while True:  # This is the main loop
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()
        dark_score = int(dark_score_s)
        light_score = int(light_score_s)

        if status == "FINAL":  # Game is over.
            print
        else:
            # Read in the input and turn it into a Python
            board = eval(input())
            # object. The format is a list of rows. The
            # squares in each row are represented by
            # 0 : empty square
            # 1 : dark disk (player 1)
            # 2 : light disk (player 2)

            # Select the move and send it to the manager
            if (minimax == 1):  # run this if the minimax flag is given
                movei, movej = select_move_minimax(
                    board, color, limit, caching)
            else:  # else run alphabeta
                movei, movej = select_move_alphabeta(
                    board, color, limit, caching, ordering)

            print("{} {}".format(movei, movej))


if __name__ == "__main__":
    run_ai()
