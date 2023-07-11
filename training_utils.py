from typing import Dict, List


def qa_split(examples: List[Dict[str, str]]) -> List[str]:
    '''Splits the examples into questions and answers.
    
    Adapts structure of output of load_FinetuningData for tokenizer.
    '''
    questions = [example["prompt"] for example in examples]
    answers = [example["target"] for example in examples]
    return questions, answers


def preprocess_data(text_pairs, tokenizer, model, max_length=512):
    prompt_text = [prompt for prompt, target in text_pairs]
    prompt_encoded = tokenizer.batch_encode_plus(
        prompt_text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='tf'
    )

    prompt_input_ids = np.array(prompt_encoded["input_ids"], dtype="int32")
    prompt_attention_masks = np.array(prompt_encoded["attention_mask"], dtype="int32")

    target_text = [target for orig, target in text_pairs]
    target_encoded = tokenizer.batch_encode_plus(
        target_text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

    label_ids = np.array(target_encoded['input_ids'])
    decoder_input_ids = model._shift_right(label_ids)

    return [prompt_input_ids, prompt_attention_masks, decoder_input_ids], label_ids


class MultihopQADataGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 tokenizer,
                 model,
                 n_examples,
                 data_filename,
                 max_length=512,
                 batch_size=16,
                 shuffle=True):

        self.tokenizer = tokenizer
        self.model = model
        self.n_examples = n_examples
        self.data_filename = data_filename
        self.max_length = max_length
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Initialize row order, call on_epoch_end to shuffle row indices
        self.row_order = np.arange(1, self.n_examples+1)
        self.on_epoch_end()

    def __len__(self):
        # Return the number of batches in the full dataset
        return self.n_examples // self.batch_size

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = (idx + 1) * self.batch_size

        # Indices to skip are the ones in the shuffled row_order before and
        # after the chunk we'll use for this batch
        batch_idx_skip = self.row_order[:batch_start] + self.row_order[batch_end:]
        df = pd.read_json(self.data_filename)

        text_pairs = df[['prompt', 'target']].values.astype(str).tolist()

        batch_data = preprocess_data(
            text_pairs,
            self.tokenizer,
            self.model,
            self.max_length
        )

        return batch_data

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

            if i == self.__len__()-1:
                self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            self.row_order = list(np.random.permutation(self.row_order))