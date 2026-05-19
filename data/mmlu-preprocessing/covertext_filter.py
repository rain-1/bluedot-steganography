from datasets import load_dataset

dataset_name = 'TIGER-Lab/MMLU-Pro'
split_name = 'validation' # validation 70 rows, change to test for 12k rows

# columns are:
# - question
# - options
# - answer
# - category
# and some more

def keep_highly_rated(row):
    return row['category'] == 'health'

dataset = load_dataset(dataset_name, split=split_name)
filtered_dataset = dataset.filter(keep_highly_rated)
filtered_dataset.save_to_disk("outputs/MMLU-Pro-filtered")
