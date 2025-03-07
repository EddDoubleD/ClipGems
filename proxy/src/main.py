import clip
import torch
import uvicorn
import logging

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from gem_repository import GemRepository, GemRepositoryConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

logging.basicConfig(
    level=logging.INFO,
    filename="full.log",
    filemode="w",
    format='%(asctime)s %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

logger = logging.getLogger(__name__)

app = FastAPI()

# Подключение статических файлов и шаблонов
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
try:
    config = GemRepositoryConfig()
    repo: GemRepository = GemRepository(
        config=config,
        collection="gems"
    )
except Exception as e:
    logger.error(f"Failed to create GemRepository: {e}")
    exit(1)

# Mock data для демонстрации
gems = [
    {
        "thumbnail": "clip-gems-income/stok/1999341160.png",
        "original": "path/to/original/content1",
        "metadata": {"title": "Sample Gem 1", "description": "Description of Gem 1"}
    },
    # Добавьте другие пины
]


@app.get("/home")
async def read_home(request: Request, search: str = ""):
    # Фильтрация по поисковому запросу (если требуется)
    filtered_gems = [gem for gem in gems if search.lower() in gem['metadata']['title'].lower()] if search else gems
    return templates.TemplateResponse("home.html", {"request": request, "gems": filtered_gems})


@app.get("/search")
async def read_home(request: str):
    text_input = clip.tokenize([request]).to(device)
    with torch.no_grad():
        features = model.encode_text(text_input)
        vector = normalize_vector(features)
        float_embedding = list()
        for x in vector.tolist()[0]:
            float_embedding.append(float(x))

        try:
            response = repo.search(float_embedding, out=["bucket", "metadata"])
            return response[0]
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise HTTPException(status_code=500, detail="Search error")


def normalize_vector(vector):
    """
    Takes a vector and returns its normalized version

    :param vector: original vector of arbitrary length
    :return: vector of the same direction, with length 1
    """
    return vector / torch.linalg.norm(vector)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)
