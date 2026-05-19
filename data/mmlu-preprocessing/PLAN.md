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

We *heavily* restricted: Final count: 908
This is OK for our purposes but something to keep in mind.

# Problem with MMLU Pro

Just as an investigation step I shuffled questions separate from answers. The scores went from 50% to 30% and random chance should give 13%.

>  My interpretation: 0.328 on the shuffled question/control dataset is far above the uniform random baseline of 0.1305, so the model is not mainly using the question to answer. It is extracting signal from the answer set itself.
>
>  Likely reasons:
>
>  - Some option groups contain one answer that is much more specific, canonical, or exam-like than the others.
>  - MMLU-Pro distractors may not be equally plausible after your covertext filtering.
>  - The correct answer may have stylistic artifacts: length, wording, hedging, specificity, or relation to the source generation process.
>  - Because the shuffled dataset keeps the original correct label for the answer group, this is effectively an “answer-only” baseline.
>
>  Pragmatically, this means the plain MMLU eval is contaminated by answer-only cues. Before using it as a steg covertext benchmark, I’d compare:
>
>  normal accuracy
>  shuffled-question accuracy
>  random chance accuracy
>
>  The useful question-conditioned signal is closer to normal - shuffled, not normal - random.

This is a good baseline to have in mind. But it's also not clear how the model would be doing this. Perhaps this qwen one is trained on test a little?

