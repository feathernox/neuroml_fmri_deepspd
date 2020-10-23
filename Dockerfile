FROM nvidia/cuda:10.2-devel-ubuntu18.04

RUN apt-get update && yes|apt-get upgrade
RUN apt-get install -y emacs
RUN apt-get install -y wget bzip2
RUN apt-get -y install sudo

# Add user ubuntu with no password, add to sudo group
RUN adduser --disabled-password --gecos '' ubuntu
RUN adduser ubuntu sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ubuntu
WORKDIR /home/ubuntu/
RUN chmod a+rwx /home/ubuntu/

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH /home/ubuntu/miniconda3/bin:$PATH

RUN sudo apt-get update && \
    sudo apt-get upgrade -y && \
    sudo apt-get install -y git && \
    sudo apt-get clean

# Create the environment:
RUN conda update conda
RUN conda update --all

COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# The code to run when container is started:
COPY notebooks fmri_deepspd/notebooks/
COPY src fmri_deepspd/src/

WORKDIR /home/ubuntu/fmri_deepspd
ENV PYTHONPATH "${PYTHONPATH}:/home/ubuntu/fmri_deepspd"
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

RUN ls /home/ubuntu/fmri_deepspd/notebooks/

ENTRYPOINT ["conda", "run", "-n", "myenv", "python", \
            "/home/ubuntu/fmri_deepspd/notebooks/train_multiprocessing.py"]
# CMD ["nvidia-smi"]

