package main

import (
	"bufio"
	"github.com/lucasmaystre/gokick/base"
	"github.com/lucasmaystre/gokick/kern"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"
	"log"
)

type Args struct {
	Path        string
	NIters      int
	NWorkers    int
}

func parseArgs() Args {
	nIters := flag.Int("n-iters", 10, "number of iterations")
	nWorkers := flag.Int("n-workers", 2, "number of workers")
	flag.Parse()
	if flag.NArg() != 1 {
		log.Fatal("required: path to dataset")
	}
	return Args{
		Path:       flag.Arg(0),
		NIters:     *nIters,
		NWorkers:   *nWorkers,
	}
}

type Datum struct {
	T      float64
	Black  string
	White  string
	Winner string
}

func parseData(path string) (data []Datum, items map[string]bool) {
	f, _ := os.Open(path)
	defer f.Close()

	// Create a new scanner.
	scanner := bufio.NewScanner(f)

	data = make([]Datum, 0, 10)
	items = make(map[string]bool)

	for scanner.Scan() {
		var datum Datum
		err := json.Unmarshal(scanner.Bytes(), &datum)
		if err != nil {
			panic(err)
		}
		items[datum.Black] = true
		items[datum.White] = true
		data = append(data, datum)
	}
	return
}

func main() {
	args := parseArgs()
	data, items := parseData(args.Path)

	model := base.NewTernaryModel(0.55361)

	kernel := kern.NewConstant(0.36381)
	for item, _ := range items {
		model.AddItem(item, kernel)
	}

	for _, datum := range data {
		if datum.Winner == "black" {
			model.Observe(
				map[string]float64{datum.Black: 1.0},
				map[string]float64{datum.White: 1.0},
				datum.T, false)
		} else if datum.Winner == "white" {
			model.Observe(
				map[string]float64{datum.White: 1.0},
				map[string]float64{datum.Black: 1.0},
				datum.T, false)
		} else {
			model.Observe(
				map[string]float64{datum.Black: 1.0},
				map[string]float64{datum.White: 1.0},
				datum.T, true)
		}
	}

	fmt.Printf("num. data: %v\n", len(data))
	fmt.Printf("num. workers: %v\n", args.NWorkers)
	fmt.Printf("num. iters: %v\n", args.NIters)
	start := time.Now()
	model.Fit(1.0, args.NWorkers, args.NIters, true)
	end := time.Now()
	fmt.Printf("Took %.3f sec per iteration.\n",
		end.Sub(start).Seconds() / float64(args.NIters))
}
