FROM hdgigante/python-opencv:4.11.0-ubuntu

# Create app directory
WORKDIR /app

# Update system packages
RUN apt-get update && apt-get install python3.12-venv git ffmpeg libsm6 libxext6  -y
# Install app dependencies
COPY requirements.txt ./

RUN python3 -m venv .venv && \
    . ./.venv/bin/activate && \
    python3 -m pip install ftfy regex tqdm && \
    # python3 -m pip install torch==2.3.1+cpu torchvision==0.18.1+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    python3 -m pip install git+https://github.com/openai/CLIP.git && \
    # python3 -m pip install openai-clip && \
    python3 -m pip install -r requirements.txt

# Bundle app source
COPY src /app
COPY run.sh /app

EXPOSE 8080
CMD [ "bash", "run.sh" ]