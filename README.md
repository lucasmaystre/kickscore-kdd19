# Reproducing the results of the kickscore paper

This repository contains the code to reproduce the results described in the paper

> Lucas Maystre, Victor Kristof, Matthias Grossglauser, [Pairwise Comparisons
> with Flexible Time-dynamics][1], KDD 2019.

## Instructions

Start by clone the Git repository:

    git clone https://github.com/lucasmaystre/kickscore-kdd19.git

Next, download the data from [Zenodo][2]. Extract the contents of the archive
inside the top-level directory of the repository, using

    tar xvf kickscore-kdd-20190725.tar.gz

Set up an environment variable pointing to the data folder:

    export KSEVAL_DATASETS=path/to/data

You might want to put this last line in your `~/.bashrc` to avoid having to
type it every time you open a terminal. Then, install the required Python
libraries:

    pip install -r requirements.txt

Now you can start a notebook server as follows as follows:

    cd notebooks
    jupyter notebook

[1]: https://arxiv.org/abs/1903.07746
[2]: https://zenodo.org/record/3351648
