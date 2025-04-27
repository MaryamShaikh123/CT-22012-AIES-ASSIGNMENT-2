##TIC-TAC-TOE GAME USING MINiMAX AND ALPHA BETA PRUNING ALGORITHM

##CT-22012 CS SEC A


import numpy as np
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1

    def make_move(self, row, col):
        if self.board[row][col] == 0:
            self.board[row][col] = self.current_player
            self.current_player = 3 - self.current_player
            return True
        return False

    def check_winner(self):
        # Check rows and columns
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != 0:
                return self.board[i][0]
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != 0:
                return self.board[0][i]

        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            return self.board[0][2]

        # Check for draw
        if np.all(self.board != 0):
            return 0

        return None

    def get_available_moves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == 0]

    def print_board(self):
        symbols = {0: '.', 1: 'X', 2: 'O'}
        for row in self.board:
            print(' '.join(symbols[cell] for cell in row))
        print()

class AIPlayer:
    def __init__(self, player_num, algorithm='alphabeta'):
        self.player_num = player_num
        self.algorithm = algorithm
        self.nodes_evaluated = 0
        self.transposition_table = {}

    def reset_counter(self):
        self.nodes_evaluated = 0
        self.transposition_table = {}

    def get_move(self, game):
        self.reset_counter()
        best_move = None
        best_value = -float('inf')

        start_time = time.time()

        for move in game.get_available_moves():
            new_game = deepcopy(game)
            new_game.make_move(*move)

            if self.algorithm == 'minimax':
                value = self.minimax(new_game, False)
            else:  # alphabeta
                value = self.alpha_beta(new_game, False, -float('inf'), float('inf'))

            if value > best_value:
                best_value = value
                best_move = move

        elapsed = time.time() - start_time
        return best_move, self.nodes_evaluated, elapsed

    def minimax(self, game, is_maximizing):
        self.nodes_evaluated += 1
        winner = game.check_winner()

        if winner is not None:
            if winner == self.player_num:
                return 1
            elif winner == 0:
                return 0
            else:
                return -1

        if is_maximizing:
            best_value = -float('inf')
            for move in game.get_available_moves():
                new_game = deepcopy(game)
                new_game.make_move(*move)
                value = self.minimax(new_game, False)
                best_value = max(best_value, value)
            return best_value
        else:
            best_value = float('inf')
            for move in game.get_available_moves():
                new_game = deepcopy(game)
                new_game.make_move(*move)
                value = self.minimax(new_game, True)
                best_value = min(best_value, value)
            return best_value

    def alpha_beta(self, game, is_maximizing, alpha, beta):
        self.nodes_evaluated += 1
        winner = game.check_winner()

        if winner is not None:
            if winner == self.player_num:
                return 1
            elif winner == 0:
                return 0
            else:
                return -1

        if is_maximizing:
            value = -float('inf')
            for move in game.get_available_moves():
                new_game = deepcopy(game)
                new_game.make_move(*move)
                value = max(value, self.alpha_beta(new_game, False, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float('inf')
            for move in game.get_available_moves():
                new_game = deepcopy(game)
                new_game.make_move(*move)
                value = min(value, self.alpha_beta(new_game, True, alpha, beta))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

def compare_algorithms():
    print("\n1. ALGORITHM COMPARISON ON DIFFERENT BOARD STATES")
    print("=" * 50)

    test_positions = [
        [],  # Empty board
        [(0,0), (1,1)],  # X in corner, O in center
        [(0,0), (0,1), (1,1)],  # X in two corners, O in center
        [(0,0), (1,1), (0,1), (1,0)],  # More complex
        [(0,0), (0,1), (1,1), (1,2), (2,2)],  # Near win scenario
        [(0,0), (1,0), (0,1), (1,1), (0,2)]  # X about to win
    ]

    results = []

    for i, position in enumerate(test_positions):
        game = TicTacToe()
        for move in position:
            game.make_move(*move)

        print(f"\nTest Case {i+1}:")
        game.print_board()

        # Test Minimax
        ai_minimax = AIPlayer(game.current_player, algorithm='minimax')
        minimax_move, minimax_calls, minimax_time = ai_minimax.get_move(game)

        # Test Alpha-Beta
        ai_alphabeta = AIPlayer(game.current_player, algorithm='alphabeta')
        alphabeta_move, alphabeta_calls, alphabeta_time = ai_alphabeta.get_move(game)

        # Calculate reductions
        call_reduction = (1 - (alphabeta_calls / minimax_calls)) * 100 if minimax_calls > 0 else 0
        time_reduction = (1 - (alphabeta_time / minimax_time)) * 100 if minimax_time > 0 else 0

        results.append({
            'test_case': i+1,
            'minimax': {'calls': minimax_calls, 'time': minimax_time * 1000},  # convert to ms
            'alpha_beta': {'calls': alphabeta_calls, 'time': alphabeta_time * 1000},
            'call_reduction': call_reduction,
            'time_reduction': time_reduction
        })

        print(f"Minimax - Calls: {minimax_calls}, Time: {minimax_time*1000:.2f}ms")
        print(f"AlphaBeta - Calls: {alphabeta_calls}, Time: {alphabeta_time*1000:.2f}ms")
        print(f"Reduction - Calls: {call_reduction:.1f}%, Time: {time_reduction:.1f}%")

    print("\n2. PERFORMANCE VISUALIZATION")
    print("=" * 50)

    test_cases = [f"Test {r['test_case']}" for r in results]
    minimax_calls = [r['minimax']['calls'] for r in results]
    alphabeta_calls = [r['alpha_beta']['calls'] for r in results]
    minimax_times = [r['minimax']['time'] for r in results]
    alphabeta_times = [r['alpha_beta']['time'] for r in results]
    call_reductions = [r['call_reduction'] for r in results]
    time_reductions = [r['time_reduction'] for r in results]

    df_calls = pd.DataFrame({
        'Test Case': test_cases,
        'Minimax': minimax_calls,
        'Alpha-Beta': alphabeta_calls
    })

    df_times = pd.DataFrame({
        'Test Case': test_cases,
        'Minimax': minimax_times,
        'Alpha-Beta': alphabeta_times
    })

    plt.figure(figsize=(12, 6))
    df_calls_melted = pd.melt(df_calls, id_vars=['Test Case'], var_name='Algorithm', value_name='Function Calls')
    sns.barplot(x='Test Case', y='Function Calls', hue='Algorithm', data=df_calls_melted)
    plt.title('Function Calls Comparison: Minimax vs Alpha-Beta Pruning')
    plt.yscale('log')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    df_times_melted = pd.melt(df_times, id_vars=['Test Case'], var_name='Algorithm', value_name='Execution Time (ms)')
    sns.barplot(x='Test Case', y='Execution Time (ms)', hue='Algorithm', data=df_times_melted)
    plt.title('Execution Time Comparison: Minimax vs Alpha-Beta Pruning')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    df_improvements = pd.DataFrame({
        'Test Case': test_cases,
        'Function Call Reduction (%)': call_reductions,
        'Execution Time Reduction (%)': time_reductions
    })
    df_improvements_melted = pd.melt(df_improvements, id_vars=['Test Case'], var_name='Metric', value_name='Reduction (%)')
    sns.barplot(x='Test Case', y='Reduction (%)', hue='Metric', data=df_improvements_melted)
    plt.title('Performance Improvement of Alpha-Beta Pruning over Minimax')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print("\n3. ANALYSIS AND SUMMARY")
    print("=" * 50)

    avg_call_reduction = sum(call_reductions) / len(call_reductions)
    avg_time_reduction = sum(time_reductions) / len(time_reductions)

    print(f"Average Function Call Reduction: {avg_call_reduction:.2f}%")
    print(f"Average Execution Time Reduction: {avg_time_reduction:.2f}%")

    max_call_idx = call_reductions.index(max(call_reductions))
    print(f"\nHighest Function Call Reduction: {max(call_reductions):.2f}% in Test Case {max_call_idx + 1}")
    print(f"  - Minimax Calls: {minimax_calls[max_call_idx]}")
    print(f"  - Alpha-Beta Calls: {alphabeta_calls[max_call_idx]}")

    print("\n4. THEORETICAL ANALYSIS")
    print("=" * 50)
    print("Minimax Time Complexity: O(b^d)")
    print("Alpha-Beta Pruning Best Case Time Complexity: O(b^(d/2))")
    print("Where b is the branching factor (available moves) and d is the depth of the game tree")

def play_game_with_ai(algorithm='minimax'):
    print("\nChoose your side:")
    print("1. X (First player)")
    print("2. O (Second player)")
    human_player = int(input("Select side (1/2): "))

    game = TicTacToe()
    ai = AIPlayer(3 - human_player, algorithm=algorithm)

    while game.check_winner() is None:
        game.print_board()
        if game.current_player == human_player:
            print("Your move (row col, 0-based):")
            try:
                row, col = map(int, input().split())
                if not game.make_move(row, col):
                    print("Invalid move! Try again.")
            except:
                print("Invalid input! Try again.")
        else:
            print(f"AI ({algorithm}) thinking...")
            move, nodes, elapsed = ai.get_move(game)
            game.make_move(*move)
            print(f"AI plays at {move} (evaluated {nodes} nodes in {elapsed*1000:.2f}ms)")

    game.print_board()
    winner = game.check_winner()
    if winner == human_player:
        print("You win!")
    elif winner == 3 - human_player:
        print(f"AI ({algorithm}) wins!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    print("Tic-Tac-Toe AI Comparison")
    print("1. Play against AI (choose algorithm)")
    print("2. Compare algorithms (generates detailed report)")
    choice = input("Select option (1/2): ")

    if choice == '1':
        print("\nChoose AI Algorithm:")
        print("1. Minimax")
        print("2. Alpha-Beta Pruning")
        algo_choice = input("Select algorithm (1/2): ")
        algorithm = 'minimax' if algo_choice == '1' else 'alphabeta'
        play_game_with_ai(algorithm)
    elif choice == '2':
        compare_algorithms()
    else:
        print("Invalid choice")