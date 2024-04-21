from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
import os
from PIL import Image
from typing import List


def tokenize_labels(processor, text_batch, device):
  return processor(text=text_batch, padding=True, images=None, return_tensors='pt').to(device)


def acquisition_model(device: str):
  model_id = "openai/clip-vit-base-patch32"
  processor = CLIPProcessor.from_pretrained(model_id)
  model = CLIPModel.from_pretrained(model_id)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)
  return processor, model


def get_text_embeddings(list_of_text: List[str], processor, model, device):
  labels = list_of_text
  label_emb = model.get_text_features(
    **tokenize_labels(processor, labels, device))
  label_emb = label_emb.detach().cpu().numpy()
  label_emb = label_emb / np.linalg.norm(label_emb, axis=1, keepdims=True)
  return label_emb


def get_image_embeddings(list_of_image_paths: List[str], processor, model, device):
  images = [Image.open(os.path.join(img_path)).convert("RGB")
            for img_path in list_of_image_paths]
  image = processor(
      text=None,
      images=images,
      return_tensors='pt'
  )['pixel_values'].to(device)
  img_emb = model.get_image_features(image)
  img_emb = img_emb.detach().cpu().numpy()
  img_emb = img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True)
  return img_emb


def prompt_proximity_score(image_embeddings, prompt_embedding):
  """
    Raw prompt score
  """
  scores = (image_embeddings @ prompt_embedding.T)
  return scores.flatten().tolist()


def max_few_shot_score(image_embeddings, few_shot_embeddings):
  scores = (image_embeddings @ few_shot_embeddings.T)
  max_scores = np.max(scores, axis=1)
  return max_scores.flatten()
