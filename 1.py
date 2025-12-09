#!/usr/bin/env python3
import tkinter as tk
from tkinter import messagebox
import math
import random
from copy import deepcopy

# ---- Configuration ----
ROWS = 6
COLUMNS = 7
CONNECT_N = 4

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

CELL_SIZE = 80
PADDING = 5

# Heuristic weights
SCORE_FOUR = 100000
SCORE_THREE = 50
SCORE_TWO = 5
SCORE_BLOCKED_THREE = 20
CENTER_WEIGHT = 3


# ---- Board class ----
class Board:
    def __init__(self):
        self.board = [[EMPTY for _ in range(COLUMNS)] for _ in range(ROWS)]

    def clone(self):
        new_b = Board()
        new_b.board = deepcopy(self.board)
        return new_b

    def drop_piece(self, col, piece):
        for r in range(ROWS - 1, -1, -1):
            if self.board[r][col] == EMPTY:
                self.board[r][col] = piece
                return r
        return None

    def remove_piece(self, col):
        for r in range(ROWS):
            if self.board[r][col] != EMPTY:
                self.board[r][col] = EMPTY
                return True
        return False

    def get_valid_locations(self):
        return [c for c in range(COLUMNS) if self.board[0][c] == EMPTY]

    def is_full(self):
        return all(self.board[0][c] != EMPTY for c in range(COLUMNS))

    def winning_move(self, piece):
        # horizontal
        for r in range(ROWS):
            for c in range(COLUMNS - CONNECT_N + 1):
                if all(self.board[r][c + i] == piece for i in range(CONNECT_N)):
                    return True
        # vertical
        for c in range(COLUMNS):
            for r in range(ROWS - CONNECT_N + 1):
                if all(self.board[r + i][c] == piece for i in range(CONNECT_N)):
                    return True
        # positive diagonal /
        for r in range(CONNECT_N - 1, ROWS):
            for c in range(COLUMNS - CONNECT_N + 1):
                if all(self.board[r - i][c + i] == piece for i in range(CONNECT_N)):
                    return True
        # negative diagonal \
        for r in range(ROWS - CONNECT_N + 1):
            for c in range(COLUMNS - CONNECT_N + 1):
                if all(self.board[r + i][c + i] == piece for i in range(CONNECT_N)):
                    return True
        return False


# ---- Heuristic evaluation ----
def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE

    count_piece = window.count(piece)
    count_opp = window.count(opp_piece)
    count_empty = window.count(EMPTY)

    if count_piece == 4:
        score += SCORE_FOUR
    elif count_piece == 3 and count_empty == 1:
        score += SCORE_THREE
    elif count_piece == 2 and count_empty == 2:
        score += SCORE_TWO

    if count_opp == 3 and count_empty == 1:
        score -= SCORE_BLOCKED_THREE

    return score


def score_position(board, piece):
    score = 0

    # Center preference
    center_col = COLUMNS // 2
    center_count = sum(1 for r in range(ROWS) if board.board[r][center_col] == piece)
    score += center_count * CENTER_WEIGHT

    # Horizontal
    for r in range(ROWS):
        row_array = board.board[r]
        for c in range(COLUMNS - CONNECT_N + 1):
            window = row_array[c:c + CONNECT_N]
            score += evaluate_window(window, piece)

    # Vertical
    for c in range(COLUMNS):
        col_array = [board.board[r][c] for r in range(ROWS)]
        for r in range(ROWS - CONNECT_N + 1):
            window = col_array[r:r + CONNECT_N]
            score += evaluate_window(window, piece)

    # Positive diagonal
    for r in range(CONNECT_N - 1, ROWS):
        for c in range(COLUMNS - CONNECT_N + 1):
            window = [board.board[r - i][c + i] for i in range(CONNECT_N)]
            score += evaluate_window(window, piece)

    # Negative diagonal
    for r in range(ROWS - CONNECT_N + 1):
        for c in range(COLUMNS - CONNECT_N + 1):
            window = [board.board[r + i][c + i] for i in range(CONNECT_N)]
            score += evaluate_window(window, piece)

    return score


def is_terminal_node(board):
    return (
        board.winning_move(PLAYER_PIECE)
        or board.winning_move(AI_PIECE)
        or board.is_full()
    )


def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = board.get_valid_locations()
    is_terminal = is_terminal_node(board)

    if depth == 0 or is_terminal:
        if is_terminal:
            if board.winning_move(AI_PIECE):
                return None, SCORE_FOUR
            elif board.winning_move(PLAYER_PIECE):
                return None, -SCORE_FOUR
            else:
                return None, 0
        else:
            return None, score_position(board, AI_PIECE)

    if maximizingPlayer:
        value = -math.inf
        best_col = random.choice(valid_locations)
        ordered_cols = sorted(valid_locations, key=lambda c: abs(c - COLUMNS // 2))

        for col in ordered_cols:
            row = board.drop_piece(col, AI_PIECE)
            if row is None:
                continue

            _, new_score = minimax(board, depth - 1, alpha, beta, False)
            board.remove_piece(col)

            if new_score > value:
                value = new_score
                best_col = col

            alpha = max(alpha, value)
            if alpha >= beta:
                break

        return best_col, int(value)

    else:
        value = math.inf
        best_col = random.choice(valid_locations)
        ordered_cols = sorted(valid_locations, key=lambda c: abs(c - COLUMNS // 2))

        for col in ordered_cols:
            row = board.drop_piece(col, PLAYER_PIECE)
            if row is None:
                continue

            _, new_score = minimax(board, depth - 1, alpha, beta, True)
            board.remove_piece(col)

            if new_score < value:
                value = new_score
                best_col = col

            beta = min(beta, value)
            if alpha >= beta:
                break

        return best_col, int(value)


class MinimaxAgent:
    def __init__(self, depth=4):
        self.depth = depth

    def pick_best_move(self, board):
        valid_locations = board.get_valid_locations()

        # Quick win if possible
        for col in valid_locations:
            row = board.drop_piece(col, AI_PIECE)
            if row is not None:
                if board.winning_move(AI_PIECE):
                    board.remove_piece(col)
                    return col
                board.remove_piece(col)

        # Minimax
        col, _ = minimax(board, self.depth, -math.inf, math.inf, True)
        if col is None:
            return random.choice(valid_locations) if valid_locations else -1
        return col


# ---- Tkinter GUI ----
class Connect4GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Connect 4")
        self.canvas = tk.Canvas(root, width=COLUMNS * CELL_SIZE, height=ROWS * CELL_SIZE, bg="blue")
        self.canvas.pack()

        self.board = Board()
        self.ai = MinimaxAgent(depth=4)
        self.turn = PLAYER_PIECE

        self.draw_board()
        self.canvas.bind("<Button-1>", self.click_event)

        self.restart_button = tk.Button(root, text="Restart", command=self.restart_game)
        self.restart_button.pack(pady=10)

    def draw_board(self):
        self.canvas.delete("all")
        for r in range(ROWS):
            for c in range(COLUMNS):
                x1 = c * CELL_SIZE + PADDING
                y1 = r * CELL_SIZE + PADDING
                x2 = (c + 1) * CELL_SIZE - PADDING
                y2 = (r + 1) * CELL_SIZE - PADDING
                piece = self.board.board[r][c]

                color = "white"
                if piece == PLAYER_PIECE:
                    color = "red"
                elif piece == AI_PIECE:
                    color = "yellow"

                self.canvas.create_oval(x1, y1, x2, y2, fill=color)

    def click_event(self, event):
        col = event.x // CELL_SIZE

        if col < 0 or col >= COLUMNS:
            return

        if self.board.board[0][col] != EMPTY:
            return

        if self.turn == PLAYER_PIECE:
            self.board.drop_piece(col, PLAYER_PIECE)
            self.draw_board()

            if self.board.winning_move(PLAYER_PIECE):
                messagebox.showinfo("Game Over", "You Win! 🎉")
                return

            if self.board.is_full():
                messagebox.showinfo("Game Over", "Draw!")
                return

            self.turn = AI_PIECE
            self.root.after(500, self.ai_move)

    def ai_move(self):
        col = self.ai.pick_best_move(self.board)
        if col != -1:
            self.board.drop_piece(col, AI_PIECE)
            self.draw_board()

            if self.board.winning_move(AI_PIECE):
                messagebox.showinfo("Game Over", "AI Wins! 💻")
                return

            if self.board.is_full():
                messagebox.showinfo("Game Over", "Draw!")
                return

        self.turn = PLAYER_PIECE

    def restart_game(self):
        self.board = Board()
        self.turn = PLAYER_PIECE
        self.draw_board()


# ---- Main ----
if __name__ == "__main__":
    root = tk.Tk()
    app = Connect4GUI(root)
    root.mainloop()
