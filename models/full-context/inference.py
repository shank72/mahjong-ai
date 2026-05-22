import torch
import numpy as np
from nn_model import MahjongMLP

def predict_discard(raw_hand_vector, checkpoint_path="../checkpoints/best_model.pt"):
    """
    Takes a single row of features (187,) and outputs the top choices
    """
    # Initialize model topology
    model = MahjongMLP()
    
    # Load your checkpoint safely
    checkpoint = torch.load(checkpoint_path, map_checkpoint=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Convert input to tensor and add a batch dimension: shape (1, 187)
    input_tensor = torch.tensor(raw_hand_vector, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        
        # Get top 3 choices
        probs, indices = torch.topk(probabilities, 3, dim=1)
        
    return probs[0].tolist(), indices[0].tolist()

if __name__ == "__main__":
    print("[INFO] Inference pipeline script ready.")
    # Example usage:
    # dummy_hand = np.random.randn(187)
    # probabilities, tile_classes = predict_discard(dummy_hand)