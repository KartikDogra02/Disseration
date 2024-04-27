import requests

class StockFishOnlineOpponent():
    def __init__(self, depth=1):
        self.base_url = "https://stockfish.online/api/s/v2.php"
        self.depth = depth

    def choose_action(self, board_fen):
        params = {
            "fen": board_fen,
            "depth": self.depth
        }
        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                best_move = data.get("bestmove").split()[1]  # Extracting the second part of "bestmove" field
                return best_move
            else:
                print("Error:", data.get("data"))
        else:
            print("Error:", response.text)
        return None
