# Litter Detection Agent

## Prompt
Ich baue ein pydantic Agenten der Müll erkennt. der litter detector Agent hat ein Subagent der frames die über Zenoh kommen empfängt und über ein trainiertes modell auswertet. ähnlich wie in @src/interference/interference.py  . Falls müll erkannt wird soll es an ein Reactor Agent übergeben werden der dann 2 Tool ausführt. das eine Tool sendet ein Alert auf ein Zenoh topic, das zweite wird ein Command an ein Robodog senden, lass dies blank

## Auswertung

| Metric                              | Score |
|-------------------------------------|-------|
| **Tool used**                       |   Claude Code    |
| **Error Rate (0 - 4)**              |     3  |
| **Code Quality (0 - 4)**            |   4    |
| **Discrepancy from Prompt (0 - 4)** |   3   |
| **Notes**                           | hat nicht die spezifizierte interference datei gelesen sondern die detector.py, config manuell nachgearbeitet   |