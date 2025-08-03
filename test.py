import utils
import chess

if __name__ == "__main__":
    turn = 'b'
    policy_index = 2239
    fen = "rnbqk2r/p4ppp/4pn2/6B1/PppP4/4PN2/5PPP/R2QKB1R w KQkq - 0 11"
    
    board = chess.Board(fen)

    from_row, from_col, channel = utils.policy_flat_index_to_components(policy_index)
    move = utils.policy_components_to_move(from_row, from_col, channel, board)

    if move is None:
        print("Invalid or illegal move for this board.")
    else:
        print("Corresponding move:", move.uci())