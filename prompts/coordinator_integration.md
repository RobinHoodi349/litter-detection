# Coordinator / integration
## prompt
Ich habe ein Litter detection Agent gebaut der ein Robodog mit Kamera kontrolliert der Agent soll:
- der Robodg kontrollieren, und mit ihm ein Suchfeld ablaufen
  -  dazu wird erst ein pfad geplant der danach abgelaufen wird
- gleichzeitig werden die Kamera bilder ausgewertet. Ein ML-modell erknnt ob Litter darauf zu sehen ist und sendet es zur  Verifikation an ein LLM wenn Litter erkannt wird soll das Movement des Roboters geblockt werden.

**Aufgabe**
- passe die beiden Seiten an, sodass sie zusammen funktionieren
die relevanten dateien findest du in @src/litter_detection/agent/  
- zudem brauche ich ein coordinator der die beiden gleichzeitig laufen lässt, bis der gesamte bereich abgesucht ist

## Analyse
| Metric                              | Score |
|-------------------------------------|-------|
| **Tool used**                       |   Claude    |
| **Error Rate (0 - 4)**              |    3   |
| **Code Quality (0 - 4)**            |    3   |
| **Discrepancy from Prompt (0 - 4)** |    4   |
| **Notes**                           |    block command zusammenspiel war etwas zu komplex implementiert, manuell vereinfacht   |