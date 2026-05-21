# Path Planning update

## prompt

### Ausgangslage

Robodog der ein Bereich auf müll absuchen soll.

- Lidar- Mapping
- A* path planning for new camera frontiers
- camera detects litter using Ml models and LLM

### Problem

- ich vermute der Pfad wird nicht neu geplant wenn neue Lidardaten erscheinen
- Der Robodog will gegen wände Laufen

### Aufgabe

- fixe die Path-planning / explore logic, sodass der Pfad neu erstellt wird
- füge ein timeout ein, dass ein neuer Pfad geplannt wird wenn der erste nicht erricht werden kann
- die frontier logic updaten, dass nicht der nächste frontier genommen wird, sondern der nächste große (mit einer cost-function)

## Auswertung

| Metric                              | Score |
|-------------------------------------|-------|
| **Tool used**                       |Claude |
| **Error Rate (0 - 4)**              |   4   |
| **Code Quality (0 - 4)**            |  4    |
| **Discrepancy from Prompt (0 - 4)** |   4   |
| **Notes**                           |   -   |
