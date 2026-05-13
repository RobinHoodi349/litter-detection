# Litter Detection Agent
## prompt
Du bist ein erfahrener Python-Softwareentwickler mit Expertise in Computer Vision, ML-Inferenz-Pipelines und Robotik-Middlewares.

Ich baue einen Litter-Detection Agent für einen Robodog, der eine Strecke abläuft und eine Kamera montiert hat. Implementiere den vollständigen Agent als Python-Code mit folgender Architektur:

Pipeline:

1. Kamera-Feed einlesen (Frame-by-Frame)
2. Jeden Frame durch ein ML-Modell (z.B. YOLO oder ein austauschbares Objekterkennungsmodell) auf Litter analysieren
3. Bei positivem ML-Befund den Frame + Detektions-Metadaten (Bounding Boxes, Confidence-Score) an ein LLM zur Verifikation übergeben (z.B. via OpenAI Vision API oder vergleichbares)
4. Wenn das LLM Litter bestätigt:
   - Eine Benachrichtigung via Zenoh publishen (Topic und Payload sinnvoll wählen, z.B. JSON mit Position, Confidence, Timestamp)
   - Die Bewegung des Robodogs blockieren — diese Funktion soll als leerer Stub block_movement() implementiert werden mit einem # TODO-Kommentar
Anforderungen:

- Saubere Trennung der Komponenten (ML-Inferenz, LLM-Verifikation, Zenoh-Kommunikation, Movement-Control) in eigene Funktionen oder Klassen
- Fehlerbehandlung für Kamera-Zugriff, ML-Inferenz und LLM-API-Calls
- Konfigurierbare Parameter (Confidence-Threshold, Zenoh-Session-Config, Kamera-Index) über Konstanten oder ein Config-Objekt oben im Code
- Zenoh-Integration mit dem eclipse-zenoh Python SDK (import zenoh)
- Der Code soll produktionsreif und gut kommentiert sein
- Verwende async/await wo es sinnvoll ist (z.B. für den Zenoh-Publisher oder LLM-Call)

## Auswertung
| Metric                              | Score |
|-------------------------------------|-------|
| **Tool used**                       |   Claude    |
| **Error Rate (0 - 4)**              |   4    |
| **Code Quality (0 - 4)**            |   4    |
| **Discrepancy from Prompt (0 - 4)** |   4    |
| **Notes**                           |   -    |