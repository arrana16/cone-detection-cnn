# Cone Colour Classification

This context defines the meaning of the labels and crops used by the offline cone-colour classifier.

## Language

**Cone crop**: A square RGB image region centered on one annotated cone, with a side 1.3 times the longer annotation dimension. The crop is resized to 32×32 for the blue/yellow/unknown task.

**Other**: A legacy classifier label for the small and large orange source labels.

**Unknown cone**: A source annotation whose cone colour is not assigned to blue, yellow, or orange.

**Unknown**: The classifier label for source annotations of orange cones and unknown cones in the blue/yellow/unknown task.
