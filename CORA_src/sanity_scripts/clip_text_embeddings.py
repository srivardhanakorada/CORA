import torch
import open_clip

# 1. Pick CLIP model
model_name = "ViT-B-32"   # or "ViT-L-14", "ViT-H-14", etc.
pretrained = "openai"     # or "laion2b_s32b_b79k"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load model + tokenizer
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
tokenizer = open_clip.get_tokenizer(model_name)
model = model.to(device)

# 3. Your classes
classes = ["Poodle", "Beagle", "Golden Retriever", "Airplane", "Candle"]

# (Optional) Add prompt templates like CLIP does internally
templates = [f"a photo of a {c}" for c in classes]

# 4. Tokenize
text_tokens = tokenizer(templates).to(device)

# 5. Get embeddings
with torch.no_grad():
    text_embeddings = model.encode_text(text_tokens)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)  # Normalize

