import pandas as pd
import os

def count_moves_in_high_rated_puzzles_starting_player_simplified(filepath, min_rating=2200):
    """
    Opens a CSV file, and counts the total number of moves for the starting player
    in puzzles with a 'Rating' greater than the specified minimum rating.
    The count is determined by assuming alternating turns:
    - If total moves are even, half are from the starting player.
    - If total moves are odd, (total moves + 1) / 2 are from the starting player.

    Args:
        filepath (str): The path to the CSV file.
        min_rating (int): The minimum rating for puzzles to be included in the count.

    Returns:
        int: The total number of moves for the starting player in high-rated puzzles.
             Returns -1 if the file is not found or an error occurs.
    """
    total_starting_player_moves = 0
    try:
        # Read the CSV file into a pandas DataFrame, only loading 'Moves' (index 2) and 'Rating' (index 3).
        # We assume the standard Lichess puzzle CSV format where 'Moves' is the 3rd column (index 2)
        # and 'Rating' is the 4th column (index 3).
        df = pd.read_csv(filepath, usecols=[2, 3], on_bad_lines='skip', low_memory=False)

        # Rename columns to ensure consistency, as pandas might use default numeric headers or incorrect names.
        df.columns = ['Moves', 'Rating']

        # Ensure the 'Rating' column exists and is numeric
        if 'Rating' not in df.columns:
            print(f"Error: 'Rating' column not found in {filepath} after loading specified columns. Columns found: {df.columns.tolist()}")
            return -1
        
        # Convert 'Rating' column to numeric, coercing errors to NaN
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
        
        # Drop rows where 'Rating' is NaN or 'Moves' is NaN
        df.dropna(subset=['Rating', 'Moves'], inplace=True)

        # Filter the DataFrame for puzzles with 'Rating' greater than the min_rating
        high_rated_puzzles_df = df[df['Rating'] > min_rating].copy() # Use .copy() to avoid SettingWithCopyWarning

        print(f"Processing {len(high_rated_puzzles_df)} high-rated puzzles...")

        # Calculate the number of moves for each puzzle and then sum them up
        # This lambda function counts the moves for the starting player based on total moves
        high_rated_puzzles_df['StartingPlayerMoves'] = high_rated_puzzles_df['Moves'].apply(
            lambda x: (len(x.split(' ')) + 1) // 2 if len(x.split(' ')) > 0 else 0
        )
        # Note: (N + 1) // 2 correctly handles both even (e.g., (4+1)//2 = 2) and odd (e.g., (5+1)//2 = 3) cases
        # while correctly accounting for potential empty 'Moves' strings.

        total_starting_player_moves = high_rated_puzzles_df['StartingPlayerMoves'].sum()

        return total_starting_player_moves

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return -1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return -1

if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust this path based on your project structure if 'data' is not directly under parent_dir
    parent_dir = os.path.abspath(os.path.join(current_script_dir, "../../..")) 
    file_path = os.path.join(parent_dir, "data", "lichess_tactics", "lichess_db_puzzle.csv")
    
    # You can change the minimum rating here if needed
    min_rating_threshold = 2240 
    
    print(f"Counting moves for puzzles with rating > {min_rating_threshold} from file: {file_path}")
    total_moves_count = count_moves_in_high_rated_puzzles_starting_player_simplified(file_path, min_rating=min_rating_threshold)

    if total_moves_count != -1:
        print(f"\nTotal number of moves by the STARTING player in puzzles with a rating > {min_rating_threshold}: {total_moves_count}")