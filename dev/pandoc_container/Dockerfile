FROM ubuntu:24.04

RUN apt-get update
RUN apt-get install -y pandoc

# Required for pdf output
RUN apt-get install -y texlive-latex-base texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra 

ENV XDG_DATA_HOME=/usr/local/share
ARG PANDOC_DATA_DIR=$XDG_DATA_HOME/pandoc
ARG TEMPLATES_DIR=$PANDOC_DATA_DIR/templates
RUN mkdir -p $TEMPLATES_DIR

WORKDIR /data
ENTRYPOINT ["pandoc"]
