# Case 6 - Student evidence sheet

Name / group: ____________________

## Before running

1. What would you need to know before saying a heatmap explains why an image classifier made a prediction?
2. Predict the accuracy of an always-dog classifier on this evaluation setup. Why might balanced accuracy be more informative?
3. Write one difference between **correctness**, **confidence**, **localization**, and **faithfulness**.

## Prediction contract and example selection

4. Record the train, development, and reserved-test sizes and the model architecture/training rule.
5. What does strong development accuracy establish? What does it not establish about explanation quality?
6. How were the four explanation examples selected? Why is that preferable to choosing visually interesting heatmaps after inspection?

## Effective input and preprocessing

7. Compare the original photograph, center-cropped model input, and fit-and-pad diagnostic for the two selected errors.
8. Record the center-crop and fit-and-pad p(dog) values for `Persian_190`.
9. Why is this comparison informative even though it does not isolate one causal factor?

## Heatmaps and explanation target

10. What quantity is explained during the audit? Why is the target class held fixed during interventions?
11. Choose the most convincing-looking map. Write:
    - one direct observation;
    - one hypothesis;
    - one stronger claim that requires another test.
12. Why should independently normalized heatmap colors not be compared as a common attribution scale across images?

## Integrated Gradients

13. What does the IG baseline represent in this case?
14. Compare the spatial ranking or appearance of the mean-baseline and blur-baseline IG maps. What changes?
15. State the completeness relation checked by the notebook in words.
16. If the completeness residual passes tolerance, what has been established? Name two things that have **not** been established.
17. Why is a numerically difficult baseline/integration setting a finding to investigate rather than a row to hide?

## Localization

18. Record foreground attribution share and animal area share for one selected example.
19. Compute or report the enrichment value. Why is enrichment more informative than foreground overlap alone?
20. Why is the Oxford trimap not ground-truth model reasoning?

## Perturbation evidence

21. Record the target-margin drops for top-10% and bottom-10% Grad-CAM removal under mean fill.
22. Compare that ordering with blur fill. Does the conclusion stay the same?
23. What assumptions or artifacts can a pixel-replacement test introduce?
24. Why are equal pixel counts not enough to make two perturbation regions perfectly comparable?

## Model dependence

25. What happens to gradient/Grad-CAM similarity after classifier-head randomization and whole-network randomization?
26. What would it imply if explanations remained almost unchanged after destroying the learned parameters?
27. Why does passing this sanity check still not prove that an explanation is faithful?

## Shortcut challenge

28. Describe the artificial marker and why it represents a Clever Hans-style hypothesis.
29. Compare clean, aligned-marker, and swapped-marker performance for the marker-trained head.
30. Did the marker demonstrate **dominance**, **limited sensitivity**, or **neither**? Justify with numbers.
31. Why is a modest or negative shortcut result still useful evidence?

## Reserved test and final verdict

32. Record reserved-test accuracy and balanced accuracy. What claim do these metrics support?
33. Name one important explanation question that the reserved-test score does not answer.

### Final deliverable

Write a **150-200 word audit recommendation** for the pet-photo service. Include:

- the model/data/evaluation scope;
- two measured results from different audit stages;
- one piece of evidence that supports a heatmap claim and one that limits it, where observed;
- one relevant assumption or unresolved question;
- one next experiment;
- wording you would permit the product team to use when describing the heatmap.

Avoid claims such as "the model understands pets," "the heatmap proves causality," or "the model uses only the animal" unless you can support them with evidence that this case actually provides.

## Exit ticket

In two sentences, distinguish **IG completeness** from **explanation faithfulness**. Then state the single piece of evidence from this case that most changed your initial view of the heatmaps.
