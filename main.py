import chess
import chess.pgn
import random
import torch
import torch.nn as nn
import torch.optim as optim
import functools

# Define the neural network
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(64 * 12, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x

# Create a separate cache dictionary
position_cache = {}

def set_position_value(board_fen, depth, value):
    position_cache[(board_fen, depth)] = value

@functools.lru_cache(maxsize=None)
def get_position_value(board_fen, depth):
    if (board_fen, depth) in position_cache:
        return position_cache[(board_fen, depth)]
    
    board = chess.Board(board_fen)
    
    if board.is_game_over():
        if board.result() == "1-0":
            return 1.0
        elif board.result() == "0-1":
            return -1.0
        else:
            return 0.0
    
    if depth == 0:
        return evaluate_board(board)
    
    if board.turn == chess.WHITE:
        best_value = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            value = get_position_value(board.fen(), depth - 1)
            board.pop()
            best_value = max(best_value, value)
    else:
        best_value = float('inf')
        for move in board.legal_moves:
            board.push(move)
            value = get_position_value(board.fen(), depth - 1)
            board.pop()
            best_value = min(best_value, value)
    
    return best_value

def evaluate_board(board):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    value = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            if piece.color == chess.WHITE:
                value += piece_values[piece.piece_type]
            else:
                value -= piece_values[piece.piece_type]
    
    return value / 39  # Normalize to [-1, 1]

def board_to_input(board):
    pieces_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    input_vector = torch.zeros(64 * 12)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            input_vector[i * 12 + pieces_map[piece.symbol()]] = 1
    return input_vector

def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return get_position_value(board.fen(), 0)
    
    if maximizing_player:
        max_eval = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def get_best_move(board, model, minimax_depth):
    best_move = None
    best_value = -float('inf') if board.turn == chess.WHITE else float('inf')
    
    for move in board.legal_moves:
        board.push(move)
        if board.turn == chess.WHITE:
            value = minimax(board, minimax_depth - 1, -float('inf'), float('inf'), False)
        else:
            value = minimax(board, minimax_depth - 1, -float('inf'), float('inf'), True)
        board.pop()
        
        if board.turn == chess.WHITE and value > best_value:
            best_value = value
            best_move = move
        elif board.turn == chess.BLACK and value < best_value:
            best_value = value
            best_move = move
    
    return best_move

def play_game(model, minimax_depth):
    board = chess.Board()
    game = chess.pgn.Game()
    node = game
    
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = get_best_move(board, model, minimax_depth)
        else:
            move = get_best_move(board, model, minimax_depth)
        
        board.push(move)
        node = node.add_variation(move)
    
    result = board.result()
    if result == "1-0":
        return 1, game
    elif result == "0-1":
        return -1, game
    else:
        return 0, game

def train_model(model, optimizer, board, target):
    optimizer.zero_grad()
    board_input = board_to_input(board)
    output = model(board_input)
    loss = nn.MSELoss()(output, torch.tensor([target], dtype=torch.float32))
    loss.backward()
    optimizer.step()

def play_and_train(model, optimizer, num_games, minimax_depth):
    results = []
    for i in range(num_games):
        result, game = play_game(model, minimax_depth)
        results.append(result)
        
        # Train on the game
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            train_model(model, optimizer, board, result)
        
        if (i + 1) % 10 == 0:
            print(f"Game {i + 1}/{num_games}: {'White' if result == 1 else 'Black' if result == -1 else 'Draw'} wins")
    
    return results, game if result == 1 else None, game if result == -1 else None

# Main execution
if __name__ == "__main__":
    model = ChessNet()
    optimizer = optim.Adam(model.parameters())
    
    num_games = 10
    minimax_depth = 5
    
    model_path = "chess_model_vs_minimax.pth"
    
    try:
        print("Loading existing model from", model_path)
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No existing model found. Creating a new model.")
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print("Creating a new model.")
        model = ChessNet()  # Reinitialize the model
        optimizer = optim.Adam(model.parameters())
    
    # Play games and train the model
    final_results, last_win_pgn, last_loss_pgn = play_and_train(model, optimizer, num_games, minimax_depth)
    
    print("Final results:")
    print(f"Model wins: {final_results.count(1)}")
    print(f"Model losses: {final_results.count(-1)}")
    print(f"Draws: {final_results.count(0)}")
    
    # Save the model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save the last won and lost games in PGN format
    if last_win_pgn:
        with open("last_win.pgn", "w") as f:
            print(last_win_pgn, file=f, end="\n\n")
        print("Last won game saved to last_win.pgn")
    
    if last_loss_pgn:
        with open("last_loss.pgn", "w") as f:
            print(last_loss_pgn, file=f, end="\n\n")
        print("Last lost game saved to last_loss.pgn")
