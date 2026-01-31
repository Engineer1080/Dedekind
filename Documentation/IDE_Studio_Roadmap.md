# Fourier — IDE-Integration und Fourier Studio (Wissenschaftler-IDE)

**Fourier Language**  
Draft: January 2026

---

## 1. Vision und Ziele

Zwei Säulen sollen parallel verfolgt werden:

1. **Fourier nahtlos in bestehenden IDEs** — VS Code, Cursor, Jupyter — um **Beliebtheit und Reichweite** zu erhöhen. Nutzer können Fourier dort einsetzen, wo sie ohnehin arbeiten.
2. **Fourier Studio als überlegene IDE für Wissenschaftler** — mit Einheitenchecks, Plots, nativer Postgres-Einbindung, LaTeX-Export und **agentischer KI auf Basis lokal ausgeführter Modelle** (Datenschutz, Kosten). Fourier Studio ist **kommerziell** und finanziert das Gesamtprojekt (Open-Source-Compiler, Sprache, Community).

**Leitprinzip**: Der **Fourier-Compiler und die Sprache bleiben Open Source**; Fourier Studio ist das kostenpflichtige Produkt, das Wissenschaftlern die beste UX bietet und das Projekt trägt.

---

## 2. Falls Fork: sinnvolle Kandidaten

Wenn Fourier Studio **nicht bei Null** starten soll, sind folgende **Fork-Kandidaten** realistisch (nach Aufwand/Nutzen):

| Kandidat | Beschreibung | Pro | Contra |
|----------|--------------|-----|--------|
| **Eclipse Theia** | VS-Code-kompatible IDE (TypeScript), explizit zum **Forken und Einbetten** gedacht. | Gleiche Extension-API wie VS Code; weniger Monolith; Cloud-/Desktop-Builds. | Trotzdem großes Repo; eigene Releases/Upstream-Merge. |
| **JupyterLab** | Notebook-zentrierte IDE; ihr plant eh einen Fourier-Kernel. | Wissenschaftler kennen es; Plugins (Kernel, MIME). | Stärker Notebook- als Skript-orientiert; „Fourier Lab“ wäre eher ein angepasstes JupyterLab. |
| **Spyder** | Schlanke wissenschaftliche IDE (Python, Qt). | Klein, domänennah (Plots, Variablen-Explorer). | Python-zentrisch; Fourier müsste als „Kernel“/Backend angebunden werden; UI in Python/Qt. |
| **RStudio** | Open-Source-IDE für R (C++/Qt + Electron-ähnliche Teile). | Bewährtes Wissenschafts-IDE-Layout (Konsole, Plots, Environment). | Sehr R-spezifisch; Fork wäre eher **Inspiration** (Layout, Features) als wörtlicher Code-Fork. |

**Empfehlung**:  
- **Theia** eignet sich, wenn ihr eine „VS-Code-artige“ eigene IDE (Desktop/Web) wollt und den Merge-/Release-Aufwand eingeht.  
- **JupyterLab** eignet sich, wenn „Fourier in Notebooks“ im Vordergrund steht und ihr das UI nur anpassen, nicht neu bauen wollt.  
- **Kein voller Fork**: Statt eine komplette IDE zu forken, kann Fourier Studio einen **Editor-Kern** (z. B. **Monaco** oder **CodeMirror 6**) einbetten und den Rest (Panels, Run, Plots, Postgres, KI) selbst bauen. Dann startet ihr „bei Null“ nur bei der IDE-Logik, nicht beim Editor – das ist oft der beste Kompromiss.

### 2.1 Ungefähre Codegröße (Zeilen Code)

| IDE | Zeilen Code (LOC) | Quelle / Anmerkung |
|-----|-------------------|---------------------|
| **RStudio** | **~802.000** | [Open Hub](https://openhub.net/p/r_studio) (Stand 2025); Java ~39 %, C++ ~29 %, JavaScript ~16 %, Rest u. a. R, GWT. |
| **Eclipse Theia** | **~500.000–1.000.000+** (geschätzt) | Kein offizieller LOC-Wert; Repo ~2,9 GB (git), >8.000 Commits; TypeScript/Node. |
| **JupyterLab** | **~200.000–400.000** (geschätzt) | Kein offizieller LOC-Wert; Monorepo, >26.000 Commits; überwiegend TypeScript, dazu Python. |
| **Spyder** | **~100.000–200.000** (geschätzt) | Kein offizieller LOC-Wert; >31.000 Commits; überwiegend Python (Qt). |

**Eigenes Nachmessen:** Exakte Zahlen z. B. mit [cloc](https://github.com/AlDanial/cloc) nach Klonen des Repos: `cloc .` im Repo-Root. Die Schätzungen für Theia/JupyterLab/Spyder basieren auf Repo-Größe, Commit-Zahl und typischen LOC-Verhältnissen.

### 2.2 Modernität der Codebasis

| IDE | Stack / Architektur | Einschätzung |
|-----|---------------------|--------------|
| **Eclipse Theia** | TypeScript, Monaco, LSP, DAP, modular; Cloud + Desktop aus einer Codebasis; VS-Code-Extension-kompatibel. | **Am modernsten.** Von Anfang an als IDE-Framework (ca. 2017+) mit aktuellem Web-Stack gebaut. |
| **JupyterLab** | TypeScript, React (via Lumino), Extension-basiert, Monorepo. | **Modern.** TypeScript-Frontend, klare Extension-API; etwas Erbe aus Notebook/IPython-Zeit. |
| **Spyder** | Python, Qt (QtPy/PyQt/PySide), Plugin-Architektur. | **Bewährt, aber älterer Stack.** Klassisches Desktop-UI (Qt), kein Web-Frontend; Spyder 5 hat modernisiert, bleibt aber Python/Qt. |
| **RStudio** | Java + GWT (zu JS kompiliert), C++-Backend. | **Am wenigsten modern.** GWT (ca. 2006), großer Legacy-Anteil; UI in Java, kompiliert nach JS – heute unüblich. |

**Fazit:** Für eine **moderne Codebasis** sind **Theia** und **JupyterLab** am besten geeignet (TypeScript, aktuelle Toolchains, Extension-Ökosystem). **Spyder** ist solide, aber Desktop-only. **RStudio** lohnt sich eher als **konzeptionelles Vorbild** (Layout, Wissenschafts-Features), nicht als Code-Fork wegen des veralteten Stacks.

---

## 3. Zwei Säulen im Überblick

| Säule | Ziel | Zielgruppe | Monetarisierung |
|-------|------|------------|-----------------|
| **Integration in bestehende IDEs** | Fourier überall nutzbar; mehr Nutzer, mehr Sichtbarkeit. | Alle, die VS Code/Cursor/Jupyter nutzen. | Indirekt: mehr Adoption → mehr Abos für Fourier Studio. |
| **Fourier Studio** | Premium-IDE für Wissenschaftler: Einheiten, Plots, Postgres, LaTeX, lokale KI. | Forscher, Labore, Datenwissenschaftler (Datenschutz/Kosten-sensibel). | **Kommerziell**: Abo oder Lizenz; Finanzierung des Projekts. |

---

## 4. Phase 1: Fourier in bestehenden IDEs (Reichweite)

**Ziel**: Fourier in VS Code, Cursor und Jupyter nahtlos nutzbar machen; keine eigene IDE nötig, um Fourier zu lernen oder einzusetzen.

### 4.1 VS Code / Cursor Extension

| Schritt | Inhalt | Aufwand (grober Richtwert) |
|---------|--------|----------------------------|
| **Sprach-Support** | Syntax-Highlighting für `.fourier`; ggf. TextMate-Grammatik aus bestehendem Lexer. | 1–2 Wochen |
| **Run & Debug** | „Fourier ausführen“ über Compiler-Aufruf; Ausgabe in integrierter Konsole. Optional: Debug-Adapter (Breakpoints, Variablen). | 2–4 Wochen |
| **Einheiten-Hinweise** | LSP-ähnlich oder Inlay-Hints: angezeigte Einheit von Ausdrücken (z. B. `1[m] + 2[m]` → `3[m]`). Nutzung des bestehenden `units_checker` / Compiler. | 2–3 Wochen |
| **Fehlermeldungen** | Compiler-Fehler (inkl. Einheiten) in Problems-Liste und im Editor mit Zeile/Kontext. | 1 Woche |

**Technik**: Extension mit TypeScript/JavaScript; Aufruf des Fourier-Compilers (CLI oder Python-API); optional Language Server Protocol (LSP) für erweiterte Features.

**Erfolgskriterium**: Nutzer können in VS Code/Cursor eine `.fourier`-Datei öffnen, ausführen und Einheiten-/Compiler-Fehler direkt im Editor sehen.

### 4.2 Jupyter Kernel für Fourier

| Schritt | Inhalt | Aufwand |
|---------|--------|---------|
| **Kernel-Spec** | Jupyter-Kernel, der Fourier-Code an den Compiler übergibt und Ergebnis/Plots zurückgibt. | 1–2 Wochen |
| **Zellen-Ausführen** | Code-Zelle → Compiler → Ausführung → stdout/stderr/Plots in Notebook. | 1 Woche |
| **Variablen/State** | Persistenter Laufzeit-Kontext über Zellen hinweg (wie bei Python-Kernel). | 1–2 Wochen |

**Nutzen**: Wissenschaftler arbeiten oft in Notebooks; Fourier in Jupyter ermöglicht interaktives Experimentieren (Kinetik, Fitting, Plots) in bekannter Umgebung.

**Erfolgskriterium**: Jupyter-Notebook mit Fourier-Kernel läuft; Zellen ausführbar, Plots anzeigbar.

### 4.3 Abhängigkeiten und Reihenfolge

- **Compiler-CLI stabil**: `python -m src.compiler.compiler <file>` mit klaren Exit-Codes und Fehlerausgabe (bereits weitgehend vorhanden).
- **Optional**: Compiler-API (z. B. `compile_source`, `run`) für Extension/Kernel ohne Subprocess.
- Empfehlung: **Zuerst VS Code Extension** (größte Sichtbarkeit), **danach Jupyter Kernel** (starker Nutzen für Wissenschaftler).

---

## 5. Phase 2: Fourier Studio — Überlegene Wissenschaftler-IDE

**Ziel**: Fourier Studio wird die **Referenz-IDE für Fourier** mit Features, die in allgemeinen IDEs nicht oder nur mit Aufwand verfügbar sind. Orientierung an RStudio (domänenspezifisch, wissenschaftsnah), aber mit Fokus auf Einheiten, Datenbanken, LaTeX und **lokale KI**.

### 5.1 Feature-Übersicht (Zielbild)

| Feature | Beschreibung | Priorität |
|---------|--------------|-----------|
| **Einheitenchecks im Editor** | Live oder on-save: Einheiten-Fehler (z. B. `1[m] + 1[s]`) markieren, Inlay-Hints für Einheiten von Ausdrücken. Nutzung `units_checker` + Compiler. | Hoch |
| **Plots integriert** | Ausgabe von `plot()` und ggf. weiteren Visualisierungen in einem IDE-Panel (wie bereits in Fourier Studio); Export als Bild/PDF. | Hoch (teilweise vorhanden) |
| **Native Postgres-Einbindung** | Verbindung zu PostgreSQL: Tabellen anzeigen, Abfragen in Fourier oder SQL ausführen, Ergebnisse als Tensoren/Listen nutzen. Für Labor-Daten, Messreihen, Metadaten. | Hoch |
| **LaTeX-Export** | Auswahl von Code/Formeln → Export als LaTeX (bestehendes `export_to_latex`); in IDE als „Kopieren als LaTeX“ oder Sidebar-Vorschau. | Mittel |
| **Agentische KI (lokal)** | KI-Assistent ähnlich Cursor (Code vervollständigen, Erklären, Refaktorieren, „Fourier-Code aus Beschreibung erzeugen“), aber **auf starken lokal ausgeführten Modellen** (z. B. Ollama, LM Studio, ggml). Begründung: **Datenschutz** (kein Code in die Cloud), **Kosten** (keine API-Gebühren). | Hoch |
| **Workspace / Environment** | Variablen, Konstanten, Einheiten nach Ausführung anzeigen (ähnlich RStudio Environment). | Mittel |
| **Quarto/Markdown** | Fourier-Code-Blöcke in Markdown/Quarto-Dokumenten ausführen (optional, später). | Niedrig |

### 5.2 Agentische KI mit lokalen Modellen — Detail

- **Anforderung**: Kein zwingender Upload von Code zu externen Diensten; Nutzer mit sensiblen Daten (Pharma, Klinik, Behörden) können KI nutzen.
- **Technik**: Integration mit **Ollama**, **LM Studio** oder vergleichbaren lokalen LLM-Servern; IDE sendet nur an localhost. Modelle: z. B. Llama, Mistral, Qwen (Code-fähig).
- **Funktionen**: Code-Vervollständigung, „Erkläre diesen Block“, „Schreibe Fourier-Code für …“, Refactoring, Einheiten-Erklärungen. Optional: Hybrid (lokal Standard, Cloud optional für Nutzer, die es wünschen).
- **Abgrenzung zu Cursor**: Cursor setzt auf Cloud-Modelle; Fourier Studio positioniert sich mit **„KI ohne Daten zu verlassen“** und **kalkulierbaren Kosten** für Teams.

### 5.3 Native Postgres-Einbindung — Detail

- **Anwendung**: Labor-Daten, Messreihen, Stichproben-Metadaten in Postgres; Abfragen in der IDE, Ergebnis direkt in Fourier weiterverarbeiten (z. B. `fit`, `plot`).
- **Umsetzung**: In Fourier Studio ein **Daten-Panel** oder **Query-Editor**: Verbindung (Host, Port, DB, User, Passwort) konfigurieren; SQL ausführen oder einfache Fourier-API (z. B. `db_query("SELECT ...")`) die Resultate als Listen/Tensoren bereitstellt. Optional: Fourier-Sprache um `sql`-Blöcke oder eine `postgres`-Bibliothek erweitern (Runtime).
- **Lizenz**: PostgreSQL ist Open Source; Treiber (z. B. psycopg2, node-postgres) verfügbar.

### 5.4 Implementierungs-Reihenfolge (Phase 2)

| Stufe | Inhalt | Abhängigkeit |
|-------|--------|---------------|
| **2a** | Einheitenchecks im Editor (Fehler markieren, Hover); Plots-Panel robuster und exportierbar. | Bestehender Compiler, `units_checker` |
| **2b** | LaTeX-Export aus IDE (Button/ Kontextmenü „Als LaTeX kopieren“). | `export_to_latex` |
| **2c** | Lokale KI-Integration (Ollama/LM Studio): Chat-Panel, Code-Actions (Erklären, Vervollständigen). | Lokaler LLM-Server, API-Anbindung |
| **2d** | Postgres: Verbindungsdialog, Query-Editor, Ergebnis-Tabelle; optional Fourier-Runtime-Funktion `db_query`. | Postgres-Client in Studio-Backend |
| **2e** | Workspace/Environment-Panel; optional Quarto/Markdown. | Laufzeit-Status vom Compiler/Runtime |

---

## 6. Phase 3: Kommerzialisierung von Fourier Studio

**Ziel**: Fourier Studio ist das **kostenpflichtige Produkt**; Erlöse finanzieren Entwicklung von Compiler, Sprache, Doku und Community.

### 6.1 Lizenzmodell (Vorschlag)

| Stufe | Zielgruppe | Inhalt | Einnahme |
|-------|------------|--------|----------|
| **Kostenlos / Community** | Einsteiger, Open-Source-Projekte | Fourier Studio mit eingeschränkten Features (z. B. keine Postgres, keine KI oder nur eingeschränkte lokale KI); oder zeitlich begrenzte Vollversion. | Sichtbarkeit, Adoption |
| **Pro / Team** | Einzelne Forscher, kleine Teams | Volle Features: Einheitenchecks, Plots, Postgres, LaTeX, lokale KI; Updates; optional Support. | Abo (monatlich/jährlich) |
| **Enterprise** | Institute, Firmen, Behörden | Pro-Features + zentrales Lizenzmanagement, SSO, Compliance, priorisierter Support. | Jahreslizenz, Volumen |

**Fourier-Sprache und -Compiler** bleiben **Open Source** (MIT o. ä.); Nutzer können weiterhin in VS Code/Cursor/Jupyter ohne Fourier Studio arbeiten. Fourier Studio ist der **Premium-Arbeitsplatz** für alle, die maximale UX und integrierte Wissenschafts-Features wollen.

### 6.2 Finanzfluss

- **Einnahmen**: Fourier Studio Abos/Lizenzen.
- **Verwendung**: Entwicklung Fourier Studio (IDE, Postgres, KI-Integration); Entwicklung Fourier-Compiler/Runtime (Open Source); Doku, Beispiele, Community-Events; ggf. Stipendien für Beiträgende.
- **Transparenz**: Optional „Open Book“ oder jährlicher Bericht, welcher Anteil in Open Source fließt — stärkt Vertrauen und Community.

### 6.3 Rechtliches und Abgrenzung

- **Datenschutz**: Lokale KI als Standard dokumentieren; keine Nutzerdaten/Code an Drittanbieter ohne explizite Einwilligung (bei optionaler Cloud-KI).
- **Nutzungsbedingungen**: Klarstellen: Fourier Studio ist proprietäre Software; Fourier Language/Compiler sind Open Source und unabhängig nutzbar.
- **Kompatibilität**: VS Code Extension und Jupyter Kernel bleiben **kostenlos**, damit die Sprache weit verbreitet bleibt; Fourier Studio wirbt mit Mehrwert (Postgres, KI, integrierte Wissenschafts-Features).

---

## 7. Abhängigkeiten und Risiken

| Abhängigkeit | Mitigation |
|--------------|------------|
| **Compiler stabil und wartbar** | Open Source beibehalten; Contributors können mitwirken; kommerzielle Einnahmen fließen in Qualität. |
| **Lokale LLMs leistungsfähig genug** | Modell-Empfehlungen (z. B. Code-Llama, Qwen-Coder); optionale Cloud-KI für Nutzer, die mehr Leistung wollen. |
| **Postgres-Treiber und Sicherheit** | Bewährte Bibliotheken; Credentials nur lokal speichern; Dokumentation zu sicheren Verbindungen (SSL). |
| **Akzeptanz kostenpflichtiger IDE** | Klarer Mehrwert (Einheiten, Postgres, lokale KI); kostenlose Alternative (VS Code + Extension) bleibt; RStudio/Posit als Vorbild. |

---

## 8. Grober Zeitplan (Orientierung)

| Phase | Inhalt | Zeitrahmen (Orientierung) |
|-------|--------|----------------------------|
| **Phase 1** | VS Code Extension (Syntax, Run, Fehler); danach Jupyter Kernel. | 2–4 Monate |
| **Phase 2a–2c** | Fourier Studio: Einheiten im Editor, Plots, LaTeX, lokale KI. | 3–6 Monate |
| **Phase 2d–2e** | Postgres, Workspace-Panel; optional Quarto. | 2–4 Monate |
| **Phase 3** | Lizenzmodell, Bezahlfluss, Pro/Enterprise-Tiers; Launch von Fourier Studio kommerziell. | Parallel zu Phase 2, Launch nach 2a–2c |

---

## 9. Zusammenfassung

- **Fourier in bestehenden IDEs** (VS Code, Cursor, Jupyter) erhöht Reichweite und Beliebtheit; **Fourier Studio** wird die **Premium-IDE für Wissenschaftler** mit Einheitenchecks, Plots, Postgres, LaTeX und **agentischer KI auf lokalen Modellen**.
- **Fourier Studio ist kommerziell**; Erlöse finanzieren das Gesamtprojekt; **Sprache und Compiler bleiben Open Source**.
- Diese Roadmap ergänzt die [Commercialization_Options](Commercialization_Options.md) um die konkrete IDE- und Produktstrategie und sollte mit der [Features Implementation Roadmap](Features_Implementation_Roadmap.md) sowie der Chemie/Biologie-Roadmap abgestimmt bleiben (Compiler-Features, Einheiten, LaTeX).
