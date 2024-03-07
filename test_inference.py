


import open_clip 

import os

try:
    # Replace 'your_api_token' with your actual Hugging Face API token
    os.environ["HF_TOKEN"] = "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"
except Exception as e:
    print("Error setting HF_TOKEN:", e)



print('attempting to load the open_clip model ')
model, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
print('successful')







