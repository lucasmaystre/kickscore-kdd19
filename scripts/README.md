This folder contains 4 items:

- `cb18.go` is the Go program used to benchmark the multithreaded
  implementation of kickscore,
  [gokick](https://github.com/lucasmaystre/gokick). Run it as follows:

        go run cb18.go --n-iters 10 --n-workers 2 ../data/kdd-chess-small.txt

- `preprocess.py` is a script that processes the raw datasets into a standard
  from that is then used throughout the codebase.
- `hyper` and `eval` are folders containing code to do a grid search over
  hyperparameters and to evaluate a single model, respectively, using the
  methodology described in the paper. These tasks were parallelized using an
  [HTCondor cluster](https://research.cs.wisc.edu/htcondor/), but you can adapt
  them to your computing infrastructure.
