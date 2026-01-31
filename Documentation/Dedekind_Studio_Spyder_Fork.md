# Dedekind Studio: Spyder-Fork mit nativem Python und Dedekind

**Dedekind Language**  
Draft: January 2026

---

## 1. Ziel

**Dedekind Studio** entsteht als Fork von **Spyder** und bietet:

- **Nativ Python** – wie Spyder heute (IPython Console, Variable Explorer, Plots, Editor).
- **Nativ Dedekind** – Dedekind als gleichberechtigte Sprache: Dedekind-Konsole (Jupyter-Kernel), Ausführen von `.ddk`-Dateien, Syntax-Highlighting für Dedekind.

Eine Codebasis, eine IDE: Wissenschaftler können in derselben Umgebung Python und Dedekind nutzen (z. B. Daten in Python laden, in Dedekind modellieren/fitten).

---

## 2. Technische Grundlage

| Komponente | Quelle | Anpassung |
|------------|--------|-----------|
| **IDE-Basis** | Spyder (MIT), Fork als „Dedekind Studio“ | Branding, Standard-Kernel, evtl. Menü „Dedekind“. |
| **Python-Konsole** | spyder-kernels (bestehend) | Unverändert nutzen. |
| **Dedekind-Konsole** | Dedekind Jupyter Kernel (dieses Repo: `dedekind_jupyter_kernel/`) | In Spyder als startbarer Kernel registrieren; Nutzer wählt „Python“ oder „Dedekind“ für neue Konsolen. |
| **Dedekind-Compiler** | `src/compiler/` (DedekindLanguage) | Wird vom Dedekind-Kernel aufgerufen; für „Run .ddk file“ direkt aufrufbar. |

Spyder unterstützt bereits „Connect to an existing kernel“ und das Starten von Kernels über Jupyter-Kernel-Specs. Für **Dedekind als ersten Bürger** muss der Dedekind-Kernel:

1. Als Kernel-Spec installierbar sein (bereits umgesetzt: `dedekind_jupyter_kernel/kernelspec/`).
2. Von Spyder beim Erstellen einer neuen Konsole **wählbar** sein („Python“ vs. „Dedekind“) – dazu muss Spyder alle installierten Jupyter-Kernel anzeigen und starten können; das ist bei vielen Spyder-Installationen bereits der Fall.
3. Optional: **Standard-Kernel** für neue Konsolen auf „Dedekind“ setzbar (Dedekind-Studio-spezifische Einstellung).

---

## 3. Phasen

### Phase 1: Dedekind-Kernel in Spyder nutzbar (ohne Fork)

- Dedekind-Kernel installieren (`jupyter kernelspec install ...`, siehe `dedekind_jupyter_kernel/README.md`).
- In **unverändertem Spyder**: Neue Konsole → „Connect to an existing kernel“ oder (falls Spyder es anbietet) Kernel „Dedekind“ wählen.
- **Erfolgskriterium:** In Spyder kann eine Konsole mit Dedekind-Kernel genutzt werden; Dedekind-Code wird kompiliert und ausgeführt, State bleibt erhalten.

### Phase 2: Spyder forken → Dedekind Studio

- **Repo:** Spyder-Fork liegt im **DedekindLanguage**-Repo im Ordner **`DedekindStudio/`** (eigenes Git-Clone; zum Einrichten im Repo-Root: `git clone https://github.com/spyder-ide/spyder.git DedekindStudio`). Lizenz MIT beibehalten, keine Quellcode-Veröffentlichungspflicht für eure Änderungen.
- **Branding:** Name „Dedekind Studio“, eigene Icons/Logos, Fenstertitel, ggf. Splash-Screen.
- **Kernel-Auswahl:** Sicherstellen, dass beim Öffnen einer neuen Konsole die installierten Jupyter-Kernel (inkl. Dedekind) angezeigt werden und „Dedekind“ startbar ist.
- **Optional:** Einstellung „Standard-Kernel für neue Konsolen“ = Python oder Dedekind.

### Phase 3: .ddk-Dateien und Run

- **Editor:** Syntax-Highlighting für `.ddk` ✅ — DedekindSH in `spyder/utils/syntaxhighlighters.py`: Keywords `fn`, `return`, `if`/`else`/`for`/`while`/`in`, Kommentare `#` und `//`, Strings, Zahlen; Outline-Explorer erkennt `fn name()` als Funktion.
- **Run:** Menüpunkt „Run“ / F5 für geöffnete `.ddk`-Datei: Dedekind-Compiler aufrufen (`compile_source` + `exec` oder CLI), Ausgabe in Konsole oder in einer dedizierten Dedekind-Konsole.
- **Fehler:** Compiler-Fehler (inkl. Zeilennummer) in Problems-Liste oder im Editor anzeigen.

### Phase 4: Erweiterungen (optional)

- Variable Explorer für Dedekind-Konsole (Variablen aus dem Dedekind-Kernel-Kontext anzeigen – erfordert Kommunikation mit dem Kernel oder Nachbildung aus generiertem Python-State).
- Einheiten-Hints im Editor (z. B. Inlay „[m]“ neben Ausdrücken) über Compiler/Units-Checker.
- LaTeX-Export aus Kontextmenü für ausgewählten Dedekind-Code.

---

## 4. Abhängigkeiten und Reihenfolge

1. **Dedekind-Kernel** – erledigt (siehe `dedekind_jupyter_kernel/`).
2. **Spyder-Umgebung** – Spyder aus Repo bauen oder Paket installieren; Dedekind-Kernel in derselben Python-Umgebung (oder Kernel-Python mit Zugriff auf Dedekind-Compiler) installieren.
3. **Fork** – nach erfolgreichem Test „Dedekind in Spyder“: eigenes Repo anlegen, Spyder forken, Branding und Kernel-Defaults anpassen.
4. **.ddk-Editor und Run** – nach stabilem Fork; erfordert Kenntnis der Spyder-Editor- und Run-API.

---

## 5. Kurzfassung

- **Dedekind Studio** = Spyder-Fork mit **nativ Python** (wie bisher) und **nativ Dedekind** (Dedekind-Kernel + optional .ddk-Editor und Run).
- **Dedekind-Kernel** ist implementiert; nächster Schritt: In Spyder (oder gebautem Dedekind Studio) den Kernel „Dedekind“ starten und in einer Konsole nutzen.
- **Lizenz:** Spyder ist MIT – Fork ohne Quellcode-Veröffentlichungspflicht; nur Lizenz- und Copyright-Hinweise beibehalten.

Dieses Dokument ergänzt die [IDE_Studio_Roadmap](IDE_Studio_Roadmap.md) um die konkrete Spyder-Fork-Strategie für Dedekind Studio.
