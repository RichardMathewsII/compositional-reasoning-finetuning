QAngaroo contains two datasets: MedHop and WikiHop.

These datasets focus on multi-step (alias multi-hop) reading comprehension.

The datasets are released under CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/).

Both datasets come in a masked and in an unmasked version (for details see the paper:
https://arxiv.org/abs/1710.06481). 

A training and development set is provided in .json format.
Each sample contains
- “id”: identifying the sample
- “query”: specifying the information that should be extracted from the texts
- “answer”: correct answer to the query
- “candidates”: a list of answer candidates, each of which is mentioned in one of the
- “supports”: a list of support documents
Furthermore, the wikihop development set comes with an additional field
- “annotations”: giving threefold Mechanical Turk annotations (detailed in the paper).


There is also a test set for both WikiHop and MedHop, but this will not be released in order to preserve the integrity of test results.
If you are interested in having your model evaluated on the hidden test sets of WikiHop and Medhop, you can upload your trained model on codalab.
More information about this, including a leaderboard with the highest scoring models can be found on the website: http://qangaroo.cs.ucl.ac.uk


Do not hesitate to reach out to me in case you have any questions: j.welbl@cs.ucl.ac.uk

