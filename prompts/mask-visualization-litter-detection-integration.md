# Mask Visualization Publishing for Litter Detection Agent Integration

## Prompt
Ich habe den Litter Detection Agent erweitert, sodass erkannter Müll zusätzlich als visualisiertes Bild mit Segmentierungsmaske veröffentlicht wird.

Der Agent soll:
- Kamerabilder analysieren
- ein Segmentierungsmodell ausführen
- erkannte Müllbereiche als Maske markieren
- das Bild mit Overlay veröffentlichen
- die Visualisierung parallel zur normalen Detection publishen
- die Bilder später in einer Trash-Image-Liste oder Kameraansicht anzeigen können

Zusätzlich:
- neues Zenoh-Topic für maskierte Bilder
- Overlay soll direkt auf dem Kamerabild liegen
- Veröffentlichung nur bei bestätigter Detection
- kompatibel mit bestehendem Verifier-System

**Aufgabe**
- erweitere `config.py`
- erweitere `detector.py`
- erweitere `litter-detection-agent.py`
- publishe das Bild mit Overlay auf ein eigenes Zenoh-Topic
- die Maske soll transparent rot über dem Bild liegen
- bestehender Detection-Flow darf nicht kaputt gehen

Relevante Dateien:
- `src/litter_detection/config.py`
- `src/litter_detection/agent/detector.py`
- `src/litter_detection/agent/litter-detection-agent.py`

---

# Analyse

| Metric                              | Score |
|-------------------------------------|-------|
| **Tool used**                       | Gemini 1.5 Pro |
| **Error Rate (0 - 4)**              | 1 |
| **Code Quality (0 - 4)**            | 4 |
| **Discrepancy from Prompt (0 - 4)** | 1 |
| **Notes**                           | Overlay-Publishing sauber integriert, bestehender Verifier-Workflow blieb kompatibel, zusätzliche Topics korrekt gekapselt |