# Player settings
WHITE_PLAYER_TYPE = 'human'
BLACK_PLAYER_TYPE = 'talbotMCTS'

# Game settings
USE_GUI = True
NUM_GAMES = 5
INITIAL_FEN = 'k1r5/p2R3p/N3p3/4Pp2/6p1/5nP1/P4P1P/5K2 b HAha - 0 1'

# Model settings
TALBOT_MODEL_PATH = "/Users/User/Projects/talbot/training/supervised/v3_hqgames_mcts/model/best_chess_ai_model.pth"
TALBOT_RESBLOCKS = 20
TALBOT_TIMELIMIT = 10.0
TALBOT_CPUCT = 3.0
TALBOT_BATCHSIZE = 16