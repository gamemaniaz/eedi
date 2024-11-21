# EEDI

This repository contains our experiments for the EEDI competition, which is essentially the classify mathematical misconceptions.

# Running

The main entrypoint for the experiments would be

```sh
export PYTHONPATH=.
python eedi/run.py
```

The main flags to configure are as follows (which are combinations of the base model, the knowledge type, the encoder):

```
usage: run.py [-h] [--enable-tqdm] [-m {llama3b,llama8b,qwen7b}] [-k {none,genk,tot,rag}] [-e {bge,bge-ft,allmini,allmini-ft,marcomini,marcomini-ft,paramini,paramini-ft}] [-s SEED] [-d {train,test}] [-n SAMPLE_SIZE] [-b BATCH_SIZE]

options:
  -m {llama3b,llama8b,qwen7b}, --llm {llama3b,llama8b,qwen7b}
                        llm model name
  -k {none,genk,tot,rag}, --knowledge {none,genk,tot,rag}
                        added knowledge context
  -e {bge,bge-ft,allmini,allmini-ft,marcomini,marcomini-ft,paramini,paramini-ft}, --encoder {bge,bge-ft,allmini,allmini-ft,marcomini,marcomini-ft,paramini,paramini-ft}
                        sentence encoder
  ...
```
