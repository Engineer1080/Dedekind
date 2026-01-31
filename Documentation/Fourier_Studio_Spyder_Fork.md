# Fourier Studio: Spyder-Fork mit nativem Python und Fourier

**Fourier Language**  
Draft: January 2026

---

## 1. Ziel

**Fourier Studio** entsteht als Fork von **Spyder** und bietet:

- **Nativ Python** – wie Spyder heute (IPython Console, Variable Explorer, Plots, Editor).
- **Nativ Fourier** – Fourier als gleichberechtigte Sprache: Fourier-Konsole (Jupyter-Kernel), Ausführen von `.fourier`-Dateien, Syntax-Highlighting für Fourier.

Eine Codebasis, eine IDE: Wissenschaftler können in derselben Umgebung Python und Fourier nutzen (z. B. Daten in Python laden, in Fourier modellieren/fitten).

---

## 2. Technische Grundlage

| Komponente | Quelle | Anpassung |
|------------|--------|-----------|
| **IDE-Basis** | Spyder (MIT), Fork als „Fourier Studio“ | Branding, Standard-Kernel, evtl. Menü „Fourier“. |
| **Python-Konsole** | spyder-kernels (bestehend) | Unverändert nutzen. |
| **Fourier-Konsole** | Fourier Jupyter Kernel (dieses Repo: `fourier_jupyter_kernel/`) | In Spyder als startbarer Kernel registrieren; Nutzer wählt „Python“ oder „Fourier“ für neue Konsolen. |
| **Fourier-Compiler** | `src/compiler/` (FourierLanguage) | Wird vom Fourier-Kernel aufgerufen; für „Run .fourier file“ direkt aufrufbar. |

Spyder unterstützt bereits „Connect to an existing kernel“ und das Starten von Kernels über Jupyter-Kernel-Specs. Für **Fourier als ersten Bürger** muss der Fourier-Kernel:

1. Als Kernel-Spec installierbar sein (bereits umgesetzt: `fourier_jupyter_kernel/kernelspec/`).
2. Von Spyder beim Erstellen einer neuen Konsole **wählbar** sein („Python“ vs. „Fourier“) – dazu muss Spyder alle installierten Jupyter-Kernel anzeigen und starten können; das ist bei vielen Spyder-Installationen bereits der Fall.
3. Optional: **Standard-Kernel** für neue Konsolen auf „Fourier“ setzbar (Fourier-Studio-spezifische Einstellung).

---

## 3. Phasen

### Phase 1: Fourier-Kernel in Spyder nutzbar (ohne Fork)

- Fourier-Kernel installieren (`jupyter kernelspec install ...`, siehe `fourier_jupyter_kernel/README.md`).
- In **unverändertem Spyder**: Neue Konsole → „Connect to an existing kernel“ oder (falls Spyder es anbietet) Kernel „Fourier“ wählen.
- **Erfolgskriterium:** In Spyder kann eine Konsole mit Fourier-Kernel genutzt werden; Fourier-Code wird kompiliert und ausgeführt, State bleibt erhalten.

### Phase 2: Spyder forken → Fourier Studio

- **Repo:** Spyder-Fork liegt im **FourierLanguage**-Repo im Ordner **`SpyderFork/`** (eigenes Git-Clone; zum Einrichten im Repo-Root: `git clone https://github.com/spyder-ide/spyder.git SpyderFork`). Lizenz MIT beibehalten, keine Quellcode-Veröffentlichungspflicht für eure Änderungen.
- **Branding:** Name „Fourier Studio“, eigene Icons/Logos, Fenstertitel, ggf. Splash-Screen.
- **Kernel-Auswahl:** Sicherstellen, dass beim Öffnen einer neuen Konsole die installierten Jupyter-Kernel (inkl. Fourier) angezeigt werden und „Fourier“ startbar ist.
- **Optional:** Einstellung „Standard-Kernel für neue Konsolen“ = Python oder Fourier.

### Phase 3: .fourier-Dateien und Run

- **Editor:** Syntax-Highlighting für `.fourier` (z. B. über Spyder-Editor-Erweiterung oder bestehende Mechanik für „unbekannte“ Dateitypen mit TextMate-Grammatik).
- **Run:** Menüpunkt „Run“ / F5 für geöffnete `.fourier`-Datei: Fourier-Compiler aufrufen (`compile_source` + `exec` oder CLI), Ausgabe in Konsole oder in einer dedizierten Fourier-Konsole.
- **Fehler:** Compiler-Fehler (inkl. Zeilennummer) in Problems-Liste oder im Editor anzeigen.

### Phase 4: Erweiterungen (optional)

- Variable Explorer für Fourier-Konsole (Variablen aus dem Fourier-Kernel-Kontext anzeigen – erfordert Kommunikation mit dem Kernel oder Nachbildung aus generiertem Python-State).
- Einheiten-Hints im Editor (z. B. Inlay „[m]“ neben Ausdrücken) über Compiler/Units-Checker.
- LaTeX-Export aus Kontextmenü für ausgewählten Fourier-Code.

---

## 4. Abhängigkeiten und Reihenfolge

1. **Fourier-Kernel** – erledigt (siehe `fourier_jupyter_kernel/`).
2. **Spyder-Umgebung** – Spyder aus Repo bauen oder Paket installieren; Fourier-Kernel in derselben Python-Umgebung (oder Kernel-Python mit Zugriff auf Fourier-Compiler) installieren.
3. **Fork** – nach erfolgreichem Test „Fourier in Spyder“: eigenes Repo anlegen, Spyder forken, Branding und Kernel-Defaults anpassen.
4. **.fourier-Editor und Run** – nach stabilem Fork; erfordert Kenntnis der Spyder-Editor- und Run-API.

---

## 5. Kurzfassung

- **Fourier Studio** = Spyder-Fork mit **nativ Python** (wie bisher) und **nativ Fourier** (Fourier-Kernel + optional .fourier-Editor und Run).
- **Fourier-Kernel** ist implementiert; nächster Schritt: In Spyder (oder gebautem Fourier Studio) den Kernel „Fourier“ starten und in einer Konsole nutzen.
- **Lizenz:** Spyder ist MIT – Fork ohne Quellcode-Veröffentlichungspflicht; nur Lizenz- und Copyright-Hinweise beibehalten.

Dieses Dokument ergänzt die [IDE_Studio_Roadmap](IDE_Studio_Roadmap.md) um die konkrete Spyder-Fork-Strategie für Fourier Studio.
