# Case 7 — Student evidence sheet

Name / group: ____________________

## Before running

1. What would you need to know before saying highlighted words explain a prediction?
2. Predict whether changing attention will always, sometimes, or never change the output.
   Explain your reasoning. You may revise your answer later.

## Data and model

3. Record the training sample size, validation size, vocabulary size, and unknown-token rate.
4. Compare majority, TF-IDF, and attention-model accuracy. What does this comparison establish?
   What does it not establish about explanations?
5. Explain how a state at the word “good” might contain information about “not.”

## First example

6. Record the reference label, predicted label, probability, and highest-attention position.
7. Write one observation, one hypothesis, and one stronger claim requiring another test.
8. Compare attention with gradient magnitude and replacement effects. Are their rankings
   similar? Is a negative replacement-margin drop possible? What does it mean?
9. Check whether the highest-attention position was already unknown to the vocabulary.

## Interventions

10. What stays fixed in ID replacement? What stays fixed in the attention intervention?
11. Record attention TV, probability change, margin change and class flip for uniform attention.
12. Explain why a shuffled heatmap is not automatically a different heatmap.
13. Explain why unchanged class labels alone cannot establish unchanged model behavior.

## Aggregate and diagnostic evidence

14. Record mean and median attention–gradient correlation and the number of valid examples.
15. Compare the uniform class-flip rate with the repeated-shuffle rate. What is the unit of
    analysis? Why should 30 shuffles not be treated as 30 independent sentences?
16. How was the disagreement example selected? Why should you not generalize from it alone?
17. Inspect an incorrect prediction. Could its attention be faithful despite its wrong label?
18. Compare the handwritten negation/contrast probes. Name a conclusion they cannot support.

## Final deliverable

Write a 150–200-word recommendation for the review platform. Include the claim and scope,
two measured results, evidence supporting or limiting the highlight where observed, an
unresolved question, and one next experiment. Recommend a label or description for the UI.

## Exit ticket

In two sentences, distinguish changing an input token from changing attention while holding
contextual hidden states fixed. Then state whether your opening prediction changed and why.
