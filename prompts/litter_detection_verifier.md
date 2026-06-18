# Verifier Agent
## Prompt

### Projekt zusammenfassung

- ein robodog der ein Bereich auf Müll absucht
- Es laufen zwei tasks parallel:
  - Exploration eines Raumes aufgrund von Lidar daten
  - Auswertung von Kamerabildern durch ein ML modell, die durch ein verifier LLM überprüft werden
- Wenn Litter erkannt wird wird eine Meldung ausgegeben und auf einer visualisierung angezeigt

### Problem
- der LLM Call für den verifier funktioniert nicht
- das ganze System läuft nicht sehr flüssig auf meinem Laptop

### Aufgabe

gehe die Probleme an. erstelle ein Plan vor den änderungen.

## Evalutation

| Metric                              | Score |
|-------------------------------------|-------|
| **Tool used**                       | Claude|
| **Error Rate (0 - 4)**              |   4   |
| **Code Quality (0 - 4)**            |   4   |
| **Discrepancy from Prompt (0 - 4)** |   4   |
| **Notes**                           |       |
