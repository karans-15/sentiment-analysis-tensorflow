# Dockerfile, Image, Container
# Dockerfile -> Blueprint for building images
# Image -> Template for running containers
# Container -> Actual running process where we have our packaged process

# Executes sequentially (IMP)

# python type file specifier 
FROM python:3.8  

# add files: src, dest
ADD main.py .

# install dependancies
RUN pip install tensorflow numpy pandas

COPY amazon_baby.csv .
COPY model_checkpoint.data-00000-of-00001 .
COPY model_checkpoint.index .

# Running main.py in container terminal
CMD ["python","./main.py"]
