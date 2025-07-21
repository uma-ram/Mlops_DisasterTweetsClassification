from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch"
)
print(training_args)
# import transformers
# print(transformers.__file__)
# print(transformers.__version__)