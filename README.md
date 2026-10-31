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

#### Build windows
```bash
python src/build_windows.py     
```
This saves a master manifest to windows/windows_manifest.csv

#### 