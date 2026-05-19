The preprocessing we want to do on MMLU Pro (or any multiple choice benchmark) is as follows:

* filter for 'good steg covertext':
  - Every answer should be at least a sentence long
  - Every answer should be fairly different, not high overlap e.g. one or two changed words only
* balance it for around 50% correct and 50% incorrect answers on our baseline model
* select a small portion (N rows. TODO: specify N), since we want to reduce evaluation cost and time
