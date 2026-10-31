### wifi-csi-activity-sensing
Environment aware activity predictor using wifi CSI info


```bash
python3 -m venv .venv
```

#### Run Experiment2 data parser
```bash
python src/parser_experiment2.py
```
This saves the parsed data to intermediate_csv/room_A/room_a_manifest.csv

#### Generate validation datasets for experiment 3 (25 per class)
```bash
python src/process_experiment3.py 
```

#### Build windows
```bash
python src/build_windows.py     
```
This saves a master manifest to windows/windows_manifest.csv

### Task 1
To predict the room
```bash
python src/room_classifier.py
```

### Task 2
self-supervised contrastive learning approach called SimCLR. Instead of teaching a neural network what room or activity a CSI signal represents using labels, we first teach it to recognise which CSI signals are similar to each other by creating different augmented versions of the same signal.

Suppose you collect a CSI window of shape:
\[
X \in \mathbb{R}^{150 \times 256}
\]

This represents:
- 150 time samples.
- 256 Wi-Fi subcarriers.

```bash
python temp/task2new.py
python src/task2_ssl_representation.py
```


### Task3
```bash
python src/task3_few_shot_adaptation.py
```