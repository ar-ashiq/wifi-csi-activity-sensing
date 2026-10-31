## wifi-csi-activity-sensing
Environment aware activity predictor using wifi CSI info

#### Setup the virtual environmnent
```bash
python3 -m venv .venv
```

#### Preprocess and generate windows
1. Saves the parsed data to intermediate_csv/room_A/room_a_manifest.csv
2. Generates the training and validation data
3. Saves a master manifest to windows/windows_manifest.csv

```bash
python src/run_preprocessing_and_windowing.py
```

### Task 1: Train the room classifier
```bash
python src/room_classifier.py
```

### Task 2: Self-Supervised Contrastive Learning (SimCLR)

Explores a **self-supervised representation learning approach using SimCLR (Simple Framework for Contrastive Learning of Visual Representations)** to learn meaningful representations from Channel State Information (CSI) signals without requiring room labels during the representation learning stage.

Instead of directly training a neural network to classify CSI signals into different rooms, SimCLR learns to identify similarities between different augmented views of the same CSI signal.

#### Run Task 2

Execute the following command from the project root:

```bash
python src/task2_ssl_representation.py
```

This executes the SimCLR representation learning pipeline and the associated baseline evaluation as implemented in the script.

### Task3
```bash
python src/task3_few_shot_adaptation.py
python src/evaluate_routing_contribution.py 
python src/generate_report_figures.py
```