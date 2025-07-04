# Player settings
WHITE_PLAYER_TYPE = 'human'
BLACK_PLAYER_TYPE = 'talbotMCTS'

# Game settings
USE_GUI = True
NUM_GAMES = 1
INITIAL_FEN = 'r2q3r/pp5p/3k2p1/2pNbb2/2Bn4/Q2P4/PPP2PPP/R1B2RK1 b HQha - 0 1'

# Model settings
TALBOT_MODEL_PATH = "/Users/User/Projects/talbot/training/supervised/v4_hqgames_with_tactics_opmcts/model/chess_ai_model_epoch_20.pth"
TALBOT_RESBLOCKS = 20
TALBOT_TIMELIMIT = 10.0
TALBOT_CPUCT = 5.0
TALBOT_BATCHSIZE = 32