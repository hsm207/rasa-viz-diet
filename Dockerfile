FROM rasa/rasa:2.8.27-full

USER root

RUN apt update && \
    apt install -y git \
        make

RUN pip install altair \
    black \
    jupyterlab \
    pandas