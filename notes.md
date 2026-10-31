1. Room-invariant activity representation


I would explore two-stage few-shot adaptation.
1. Separate room adaptation from activity adaptation
When the system enters Room B:

Stage 1 – Adapt to the new room

Use the K-shot CSI captures to estimate the room-specific channel characteristics.

For example, learn a room adaptation parameter:
\[
z' = f_\theta(X) + \Delta_{\text{room}}
\]

Here:
- \(f_\theta\) is the pretrained SimCLR encoder.
- \(\Delta_{\text{room}}\) adjusts the representation for the new room.
Stage 2 – Learn the new activities

Use the labelled K-shot examples to create activity prototypes:
\[
P_c = \frac{1}{K}\sum_{i=1}^{K} f_{\theta,\Delta}(X_i)
\]

The classifier now uses the adapted representation to identify activities.

This separates the room adaptation problem from learning the new activity classes.
2. Can we use learning rates?
Yes. We can use different learning rates for different parts of the model.
Model component	Learning rate	Purpose
Pretrained encoder	Very low (e.g. 1e-5)	Preserve learned features
Room adaptation layer	Higher (e.g. 1e-3)	Adapt to new room
New activity classifier	Higher (e.g. 1e-3)	Learn new activities

These are starting values for experiments, not guaranteed optimal values.

However, there is an important distinction:
- A learning rate controls how much model parameters change.
- It does not automatically distinguish room characteristics from activity characteristics.
So simply increasing the learning rate for the new activity classifier may not solve the room interference problem.
3. Another interesting approach: Adapter-based learning
Instead of fine-tuning the entire encoder, introduce a small trainable adapter.
Pretrained SimCLR encoder

Mostly frozen

Room-specific adapter

Updated using new-room CSI data

Few-shot activity prototypes

Updated using K labelled captures

Target-room activity prediction


The advantage is that you don't need to retrain the entire network every time you enter a new room.
4. One important question about your experimental setup
Your current Task 3 protocol provides K labelled captures only for target-exclusive activities, while shared activities such as SIT, STAND and WALK have no target-room labels.

This means:
- Room adaptation can use unlabelled CSI data from the new room, if your protocol permits it.
- New activity prototypes can use the K labelled captures.
- Shared activity labels cannot be used for target-room adaptation.
I would compare these three experiments:
1. Baseline: SimCLR + Prototypical Network.
2. Few-shot fine-tuning: Fine-tune the encoder with a low learning rate using K-shot examples.
3. Adapter-based adaptation: Freeze most of the encoder, adapt the room-specific layer, and learn new activity prototypes.
My recommendation is experiment 3 as a promising research direction.

One clarification: When the system enters a new room, will it have access to some unlabelled CSI samples from that room before activity prediction, or will it only receive the K labelled examples of the new activities? This determines how we should design the room adaptation stage.