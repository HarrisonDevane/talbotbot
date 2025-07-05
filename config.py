# Player settings
WHITE_PLAYER_TYPE = 'talbot'
BLACK_PLAYER_TYPE = 'human'

# Game settings
USE_GUI = True
NUM_GAMES = 1
INITIAL_FEN = None

# Model settings
TALBOT_MODEL_PATH = "/Users/User/Projects/talbot/training/supervised/v4_hqgames_with_tactics_opmcts/model/chess_ai_model_epoch_20.pth"
TALBOT_RESBLOCKS = 20
TALBOT_CPUCT = 5.0
TALBOT_BATCHSIZE = 32

# Time settings
TALBOT_TIME_PER_MOVE = 1.0
STOCKFISH_TIME_PER_MOVE = 5.0
LEELA_TIME_PER_MOVE = 5.0