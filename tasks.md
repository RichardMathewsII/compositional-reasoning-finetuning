## Task Chunks
1. **Bamboogle Data Loader** - Implement in [data_loaders.py](/data_loaders.py)
2. **Musique Data Loader** - Implement in [data_loaders.py](/data_loaders.py)
3. **Compositional celebrities train/dev/test/split**
4. **Load dataset splits to google drive which team shares**
5. **QA Accuracy Scoring**
   1. **Class** - implements `QAAccuracyScorer` class, with a `model` parameter on initialization and a `score` method with a `dataset` argument and `num_examplars` argument and `split` argument (which dataset split), computes an accuracy of model on given dataset using in-context examples.
   2. **Compositional Celebrities**
   3. **2WikiMultiHopQA**
   4. **Musique**
   5. **Bamboogle**
6. **Compositionality Gap Function** - given model and `num_examplars`, computes the compositionality gap (using compositional celebrities)
7. **Model fine-tuning**
   1. **Small Size**
      1. (question + self-ask examplars, answer)
      2. (question + self-ask examplars, self-ask rationale + answer)
   2. **Medium Size**
      1. (question + self-ask examplars, answer)
      2. (question + self-ask examplars, self-ask rationale + answer)
   3. **Large Size**
      1. (question + self-ask examplars, answer)
      2. (question + self-ask examplars, self-ask rationale + answer)
8. **QA Accuracy Evaluation**
   1. **Compositional Celebrities**
      1. **Baseline**
         1. CoT with 4 examplars
         2. Self-ask with 4 examplars
      2. **Fine-tuned (Zero-shot)**
         1. Small Size - 1
         2. Small Size - 2
         3. Medium Size - 1
         4. Medium Size - 2
         5. Large Size - 1
         6. Large Size - 2
      3. **Fine-tuned (4-shot)**
         1. Small Size - 1
         2. Small Size - 2
         3. Medium Size - 1
         4. Medium Size - 2
         5. Large Size - 1
         6. Large Size - 2
   2. **2WikiMultiHopQA**
      1. **Baseline**
         1. CoT with 4 examplars
         2. Self-ask with 4 examplars
      2. **Fine-tuned (Zero-shot)**
         1. Small Size - 1
         2. Small Size - 2
         3. Medium Size - 1
         4.  Medium Size - 2
         5.  Large Size - 1
         6.  Large Size - 2
      3. **Fine-tuned (4-shot)**
         1. Small Size - 1
         2. Small Size - 2
         3. Medium Size - 1
         4. Medium Size - 2
         5. Large Size - 1
         6. Large Size - 2
   3. **Musique**
      1. **Baseline**
         1. CoT with 4 examplars
         2. Self-ask with 4 examplars
      2. **Fine-tuned (Zero-shot)**
         1. Small Size - 1
         2. Small Size - 2
         3. Medium Size - 1
         4.  Medium Size - 2
         5.  Large Size - 1
         6.  Large Size - 2
      3. **Fine-tuned (4-shot)**
         1. Small Size - 1
         2. Small Size - 2
         3. Medium Size - 1
         4. Medium Size - 2
         5. Large Size - 1
         6. Large Size - 2
   4. **Bamboogle**
      1. **Baseline**
         1. CoT with 4 examplars
         2. Self-ask with 4 examplars
      2. **Fine-tuned (Zero-shot)**
         1. Small Size - 1
         2. Small Size - 2
         3. Medium Size - 1
         4.  Medium Size - 2
         5.  Large Size - 1
         6.  Large Size - 2
      3. **Fine-tuned (4-shot)**
         1. Small Size - 1
         2. Small Size - 2
         3. Medium Size - 1
         4. Medium Size - 2
         5. Large Size - 1
         6. Large Size - 2
9. **Compositionality Gap Measurement**
   1. **Baseline**
      1. Self-ask with 4 examplars
   2. **Fine-tuned (Zero-shot)**
      1. Small Size - 1
      2. Small Size - 2
      3. Medium Size - 1
      4.  Medium Size - 2
      5.  Large Size - 1
      6.  Large Size - 2
   3. **Fine-tuned (4-shot)**
      1. Small Size - 1
      2. Small Size - 2
      3. Medium Size - 1
      4. Medium Size - 2
      5. Large Size - 1
      6. Large Size - 2