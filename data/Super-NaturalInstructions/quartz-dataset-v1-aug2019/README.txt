QuaRTz dataset V1 - May 2019
============================

The QuaRTz dataset V1 contains 3864 questions paired with one of 405 different background sentences (sometimes short paragraphs). There is one .jsonl file for each train/dev/test split (2696/384/784 questions respectively). A background sentence will only appear in a single split. The .txt files contain the core sentences and questions in more readable format.

Here is an example line for the .jsonl format, with whitespace added for clarity:

{"answerKey":"B",
"para_id":"QRSent-10360",
"id":"QRQA-10360-5-flip",
"question":{
  "stem":"John was looking at sunscreen at the retail store. He noticed that sunscreens that had lower SPF would offer protection that is", 
  "choices":[{"label":"A","text":"Longer"},{"label":"B","text":"Shorter"}]},
"para":"A sunscreen with a higher SPF protects the skin longer.",
"para_anno":{
  "effect_dir_sign":"MORE", 
  "cause_dir_sign":"MORE",
  "effect_prop":"protection",
  "cause_prop":"SPF",
  "cause_dir_str":"higher",
  "effect_dir_str":"longer"},
"question_anno":{
  "more_effect_dir":"longer",
  "less_effect_dir":"Shorter",
  "less_cause_prop":"spf",
  "more_effect_prop":"protection",
  "less_cause_dir":"lower",
  "less_effect_prop":"protection"}
}

Explanations of the fields:

id: Unique question id, ends with "-flip" if it's a "flipped" version of an original question
para_id: Unique paragraph id
question: Contains the question stem and answer choices 
answerKey: The label corresponding to the correct answer
para: The text of the associated background sentence (paragraph)
para_anno: Annotations related to the background sentence:
   cause_dir_sign: MORE or LESS indicating the direction of change for the "cause"
   cause_prop: Surface form associated with the cause
   cause_dir: Surface form associated with the change direction
   Same for cause -> effect for the effect property
question_anno: Annotations related to the question
   more_cause_dir: Surface form (if any) associated with the direction of chance for the cause property, in the direction of "MORE"
   less_cause_dir: Same, but for direction "LESS"
   more_cause_prop: Same, but for associated property rather than direction
   Same for cause -> effect for the effect property


