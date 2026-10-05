# Cone Colour Classification

This context defines the meaning of the labels and crops used by the offline cone-colour classifier.

## Language

**Cone crop**: An RGB image region centered on one annotated cone, with 15% context around the bounding box.

**Other**: An orange cone. The small and large orange source labels share this classifier label.

**Unknown cone**: A cone whose colour is not assigned to blue, yellow, or orange in the source annotations. It is retained for separate analysis and has no training target in the three-class task.
