import os
import numpy as np
from PIL import Image
import torch
import clip

multiple_templates = [
    "There is {} in the scene.",
    "There is the {} in the scene.",
    "a photo of {} in the scene.",
    "a photo of the {} in the scene.",
    "a photo of one {} in the scene.",
    "I took a picture of of {}.",
    "I took a picture of of my {}.",  # itap: I took a picture of
    "I took a picture of of the {}.",
    "a photo of {}.",
    "a photo of my {}.",
    "a photo of the {}.",
    "a photo of one {}.",
    "a photo of many {}.",
    "a good photo of {}.",
    "a good photo of the {}.",
    "a bad photo of {}.",
    "a bad photo of the {}.",
    "a photo of a nice {}.",
    "a photo of the nice {}.",
    "a photo of a cool {}.",
    "a photo of the cool {}.",
    "a photo of a weird {}.",
    "a photo of the weird {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of a clean {}.",
    "a photo of the clean {}.",
    "a photo of a dirty {}.",
    "a photo of the dirty {}.",
    "a bright photo of {}.",
    "a bright photo of the {}.",
    "a dark photo of {}.",
    "a dark photo of the {}.",
    "a photo of a hard to see {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of {}.",
    "a low resolution photo of the {}.",
    "a cropped photo of {}.",
    "a cropped photo of the {}.",
    "a close-up photo of {}.",
    "a close-up photo of the {}.",
    "a jpeg corrupted photo of {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of {}.",
    "a blurry photo of the {}.",
    "a pixelated photo of {}.",
    "a pixelated photo of the {}.",
    "a black and white photo of the {}.",
    "a black and white photo of {}.",
    "a plastic {}.",
    "the plastic {}.",
    "a toy {}.",
    "the toy {}.",
    "a plushie {}.",
    "the plushie {}.",
    "a cartoon {}.",
    "the cartoon {}.",
    "an embroidered {}.",
    "the embroidered {}.",
    "a painting of the {}.",
    "a painting of a {}.",
]

def match_text_to_imgs(language_instr, images_list):
    """img_feats: (Nxself.clip_feat_dim), text_feats: (1xself.clip_feat_dim)"""
    imgs_feats = get_imgs_feats(images_list)
    text_feats = get_text_feats([language_instr])
    scores = imgs_feats @ text_feats.T
    scores = scores.squeeze()
    return scores, imgs_feats, text_feats


def get_nn_img(raw_imgs, text_feats, img_feats):
    """img_feats: (Nxself.clip_feat_dim), text_feats: (1xself.clip_feat_dim)"""
    scores = img_feats @ text_feats.T
    scores = scores.squeeze()
    high_to_low_ids = np.argsort(scores).squeeze()[::-1]
    high_to_low_imgs = [raw_imgs[i] for i in high_to_low_ids]
    high_to_low_scores = np.sort(scores).squeeze()[::-1]
    return high_to_low_ids, high_to_low_imgs, high_to_low_scores


def get_img_feats(img, preprocess, clip_model):
    img_pil = Image.fromarray(np.uint8(img))
    img_in = preprocess(img_pil)[None, ...]
    with torch.no_grad():
        img_feats = clip_model.encode_image(img_in.cuda()).float()
    img_feats /= img_feats.norm(dim=-1, keepdim=True)
    img_feats = np.float32(img_feats.cpu())
    return img_feats


def get_imgs_feats(raw_imgs, preprocess, clip_model, clip_feat_dim):
    imgs_feats = np.zeros((len(raw_imgs), clip_feat_dim))
    for img_id, img in enumerate(raw_imgs):
        imgs_feats[img_id, :] = get_img_feats(img, preprocess, clip_model)
    return imgs_feats


def get_imgs_feats_batch(
    raw_imgs, preprocess, clip_model, clip_feat_dim, batch_size=64
):
    imgs_feats = np.zeros((len(raw_imgs), clip_feat_dim))
    img_batch = []
    for img_id, img in enumerate(raw_imgs):
        if img.shape[0] == 0 or img.shape[1] == 0:
            img = [[[0, 0, 0]]]
        img_pil = Image.fromarray(np.uint8(img))
        img_in = preprocess(img_pil)[None, ...]
        img_batch.append(img_in)
        if len(img_batch) == batch_size or img_id == len(raw_imgs) - 1:
            img_batch = torch.cat(img_batch, dim=0)
            with torch.no_grad():
                batch_feats = clip_model.encode_image(img_batch.cuda()).float()
            batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
            batch_feats = np.float32(batch_feats.cpu())
            imgs_feats[img_id - len(img_batch) + 1 : img_id + 1, :] = batch_feats
            img_batch = []
    return imgs_feats


def get_text_feats(in_text, clip_model, clip_feat_dim, batch_size=64):
    text_tokens = clip.tokenize(in_text).cuda()
    text_id = 0
    text_feats = np.zeros((len(in_text), clip_feat_dim), dtype=np.float32)
    while text_id < len(text_tokens):  # Batched inference.
        batch_size = min(len(in_text) - text_id, batch_size)
        text_batch = text_tokens[text_id : text_id + batch_size]
        with torch.no_grad():
            batch_feats = clip_model.encode_text(text_batch).float()
        batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
        batch_feats = np.float32(batch_feats.cpu())
        text_feats[text_id : text_id + batch_size, :] = batch_feats
        text_id += batch_size
    return text_feats

def get_text_feats_multiple_templates(in_text, clip_model, clip_feat_dim, batch_size=64):
    mul_tmp = multiple_templates.copy()
    multi_temp_landmarks_other = [
        x.format(lm) for lm in in_text for x in mul_tmp
    ]
    text_feats = get_text_feats(
        multi_temp_landmarks_other, clip_model, clip_feat_dim
    )
    # average the features
    text_feats = text_feats.reshape((-1, len(mul_tmp), text_feats.shape[-1]))
    text_feats = np.mean(text_feats, axis=1)
    return text_feats