# ClipGems
decription promt: Приложение для анализа и поиска по видео-контенту<br/>

## Arch
Used model: ViT-B/32
DataBase: standalone milvus

## Code base
db config script: common/milvus
local run
```shell
source .envrc && python ./paparazzi/src/server.py 
```

## IaC
```shell
pip freeze > requirements.txt
docker build -t edddoubled/paparazzi:v1.0 .

docker compose up -d

docker run -v $(pwd)/secret:/paparazzi/secret -p 8080:8080 edddoubled/paparazzi:v1.0
```

```shell
# Stop Milvus
$ sudo docker compose down

# Delete service data
$ sudo rm -rf volumes
```

bench
https://github.com/cvdfoundation/kinetics-dataset/blob/main/README.md

