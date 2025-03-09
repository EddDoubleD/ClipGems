import logging

import numpy as np
import clip
import torch

logging.basicConfig(level=logging.DEBUG)


class Predictor(object):
    clip_model = None
    device = None
    preprocess = None
    image_mean = None
    image_std = None

    @staticmethod
    def load():
        if Predictor.clip_model is None:
            Predictor.device = "cuda" if torch.cuda.is_available() else "cpu"
            Predictor.clip_model, _process = clip.load("ViT-B/32", device=Predictor.device)
            # transformer:  Resize(BICUBIC) -> CenterCrop(IMG_SIZE) -> ToTensor() -> Normalize(MEAN, STD)
            # more information about normalization can be found here https://github.com/openai/CLIP/issues/20
            if Predictor.preprocess is None:
                Predictor.preprocess = _process
                Predictor.image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(
                    Predictor.device
                )
                Predictor.image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(
                    Predictor.device
                )

        return Predictor.clip_model

    @staticmethod
    def mean_image(image):
        img = Predictor.preprocess(image).unsqueeze(0).to(Predictor.device)
        with torch.no_grad():
            feature = (
                Predictor.clip_model.encode_image(img).float().to(Predictor.device)
            )
            feature /= feature.norm(dim=-1, keepdim=True)
            return feature

    @staticmethod
    def mean_images(frames):
        # image = torch.tensor(np.stack(frames)).to(Predictor.device)
        # frame_embeddings = get_frame_embeddings(frames)
        # video_embedding = np.mean(frame_embeddings, axis=0)
        frames_embeddings = []
        with torch.no_grad():
            for frame in frames:
                image = Predictor.preprocess(frame).unsqueeze(0).to(Predictor.device)
                feature = (
                    Predictor.clip_model.encode_image(image).float().to(Predictor.device)
                )
                feature /= feature.norm(dim=-1, keepdim=True)
                frames_embeddings.append(feature)

            return np.mean(frames_embeddings, axis=0)


    @staticmethod
    def predict(frames, classes):
        image_input = torch.tensor(np.stack(frames)).to(Predictor.device)
        # using default clip transformer
        image_input -= Predictor.image_mean[:, None, None]
        image_input /= Predictor.image_std[:, None, None]

        text_input = torch.cat(
            [clip.tokenize(f"a photo of a {c}") for c in classes]
        ).to(Predictor.device)

        with torch.no_grad():
            image_features = (
                Predictor.clip_model.encode_image(image_input).float().to(Predictor.device)
            )

            text_features = (
                Predictor.clip_model.encode_text(text_input).float().to(Predictor.device)
            )

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        result_count = min(len(classes), 3)
        values, indices = similarity.cpu().topk(result_count, dim=-1)

        response = []

        for index, value in zip(indices, values):
            indices = index.numpy().tolist()
            response.append(
                (
                    indices,
                    list(map(lambda i: classes[i], indices)),
                    value.numpy().tolist(),
                )
            )

        return response
