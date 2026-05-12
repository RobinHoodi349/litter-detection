# Training Improvements
## Prompt: 

Du bist ein erfahrener Computer-Vision-Ingenieur mit tiefem Fachwissen in Semantic Segmentation, Transfer Learning und Modelloptimierung für Umwelt-Datensätze.

Ich trainiere ein **Litter-Detection-Modell** auf dem **TACO-Datensatz** mit einem vortrainierten **EfficientNet-B1 als Encoder in einer U-Net-Architektur**. Das Modell läuft bereits, und ich möchte das Training systematisch weiter optimieren, um die beste mögliche Performance (mIoU / Dice Score) zu erreichen.

Analysiere alle relevanten Optimierungsdimensionen für mein Setup und gib konkrete, umsetzbare Empfehlungen für:

- **Daten-seitige Optimierungen**: Augmentation-Strategien, die speziell für Litter-Detection sinnvoll sind (z. B. MixUp, Mosaic, Copy-Paste für kleine Objekte), Class Imbalance Handling (TACO ist stark unbalanciert), Oversampling-Strategien
- **Trainings-Hyperparameter**: Lernraten-Scheduling (z. B. Cosine Annealing, OneCycleLR), Batch Size, Optimizer-Wahl (AdamW vs. SGD + Momentum), Warmup-Strategien
- **Loss-Funktionen**: Welche Kombination aus Dice Loss, Focal Loss, Tversky Loss oder anderen für unbalancierte Segmentierungsdaten mit TACO am besten funktioniert und warum
- **Regularisierung & Generalisierung**: Dropout, Label Smoothing, Stochastic Depth, Weight Decay
- **Fine-Tuning-Strategie**: Wann und wie man den EfficientNet-Encoder schrittweise auftaut (Layer-wise Learning Rate Decay), anstatt ihn komplett einzufrieren oder freizugeben
- **Architekturelle Verbesserungen**: Decoder-Erweiterungen (Attention Gates, ASPP, Deep Supervision), die auf U-Net aufgebaut werden können
- **Inference-seitige Optimierungen**: Test-Time Augmentation (TTA), Modell-Ensembling

Priorisiere die Empfehlungen nach erwartetem Impact-to-Effort-Verhältnis und erkläre kurz, warum jede Maßnahme speziell für den TACO-Datensatz (heterogene Müllkategorien, Klassenungleichgewicht, variierende Hintergründe) relevant ist.

## Auswertung

| Metric                              | Score |
|-------------------------------------|-------|
| **Tool used**                       |     Claude  |
| **Error Rate (0 - 4)**              |    4   |
| **Code Quality (0 - 4)**            |    4   |
| **Discrepancy from Prompt (0 - 4)** |    4   |
| **Notes**                           |    -   |