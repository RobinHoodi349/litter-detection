# Agentsaufbau
Hier sollen die Überlegungen und Definitionen dokumentiert werden, die für die Aufgabe in betracht gezogen werden. Bezogen wird sich hier auf die Aufgabenstellung aus [dem Arbeitspaket für Labor 2](../../docs/student_task_2.md).

## Aufgabe 1
### Erste Überlegung:
```
Agent --- Tools
            |--- **Laufen**: Geradeauslaufen von Punkt A nach B.
            |--- **Drehen**: Wenn B erreicht drehen.
            |--- **Scan**: Während des laufens die Umgebung nach Müll Scannen.
                    |------ **Save & Send**: Bei erfolgreicher erkennung, Position speichern & senden.
                    |------ **Emote**: Bewegung & Sound bei erfolgreicher erkennung durchführen.
```