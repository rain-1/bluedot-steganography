## Processing

The preprocessing we want to do on MMLU Pro (or any multiple choice benchmark) is as follows:

* filter for 'good steg covertext':
  - Every answer should be at least a sentence long
  - Every answer should be fairly different, not high overlap e.g. one or two changed words only
* balance it for around 50% correct and 50% incorrect answers on our baseline model
* select a small portion (N rows. TODO: specify N), since we want to reduce evaluation cost and time


python data/mmlu-preprocessing/covertext_filter.py

## Tools

Filter and inspect

```
$ python data/mmlu-preprocessing/covertext_filter.py 
Saving the dataset (1/1 shards): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 2233.39 examples/s]
$ python utils/dataset_to_jsonl.py outputs/MMLU-Pro-filtered/ | jq
```
