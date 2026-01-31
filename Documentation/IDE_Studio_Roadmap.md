# Dedekind — IDE-Integration und Dedekind Studio (Wissenschaftler-IDE)

**Dedekind Language**  
Draft: January 2026

---

## 1. Vision und Ziele

Zwei Säulen sollen parallel verfolgt werden:

1. **Dedekind nahtlos in bestehenden IDEs** — VS Code, Cursor, Jupyter — um **Beliebtheit und Reichweite** zu erhöhen. Nutzer können Dedekind dort einsetzen, wo sie ohnehin arbeiten.
2. **Dedekind Studio als überlegene IDE für Wissenschaftler** — mit Einheitenchecks, Plots, nativer Postgres-Einbindung, LaTeX-Export und **agentischer KI auf Basis lokal ausgeführter Modelle** (Datenschutz, Kosten). Dedekind Studio ist **kommerziell** und finanziert das Gesamtprojekt (Open-Source-Compiler, Sprache, Community).

**Leitprinzip**: Der **Dedekind-Compiler und die Sprache bleiben Open Source**; Dedekind Studio ist das kostenpflichtige Produkt, das Wissenschaftlern die beste UX bietet und das Projekt trägt.

---

## 2. Falls Fork: sinnvolle Kandidaten

Wenn Dedekind Studio **nicht bei Null** starten soll, sind folgende **Fork-Kandidaten** realistisch (nach Aufwand/Nutzen):

| Kandidat | Beschreibung | Pro | Contra |
|----------|--------------|-----|--------|
| **Eclipse Theia** | VS-Code-kompatible IDE (TypeScript), explizit zum **Forken und Einbetten** gedacht. | Gleiche Extension-API wie VS Code; weniger Monolith; Cloud-/Desktop-Builds. | Trotzdem großes Repo; eigene Releases/Upstream-Merge. |
| **JupyterLab** | Notebook-zentrierte IDE; ihr plant eh einen Dedekind-Kernel. | Wissenschaftler kennen es; Plugins (Kernel, MIME). | Stärker Notebook- als Skript-orientiert; „Dedekind Lab“ wäre eher ein angepasstes JupyterLab. |
| **Spyder** | Schlanke wissenschaftliche IDE (Python, Qt). | Klein, domänennah (Plots, Variablen-Explorer). | Python-zentrisch; Dedekind müsste als „Kernel“/Backend angebunden werden; UI in Python/Qt. |
| **RStudio** | Open-Source-IDE für R (C++/Qt + Electron-ähnliche Teile). | Bewährtes Wissenschafts-IDE-Layout (Konsole, Plots, Environment). | Sehr R-spezifisch; Fork wäre eher **Inspiration** (Layout, Features) als wörtlicher Code-Fork. |

**Empfehlung**:  
- **Theia** eignet sich, wenn ihr eine „VS-Code-artige“ eigene IDE (Desktop/Web) wollt und den Merge-/Release-Aufwand eingeht.  
- **JupyterLab** eignet sich, wenn „Dedekind in Notebooks“ im Vordergrund steht und ihr das UI nur anpassen, nicht neu bauen wollt.  
- **Kein voller Fork**: Statt eine komplette IDE zu forken, kann Dedekind Studio einen **Editor-Kern** (z. B. **Monaco** oder **CodeMirror 6**) einbetten und den Rest (Panels, Run, Plots, Postgres, KI) selbst bauen. Dann startet ihr „bei Null“ nur bei der IDE-Logik, nicht beim Editor – das ist oft der beste Kompromiss.

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

### 2.3 Rechtliches: Theia forken und kommerziell nutzen

**Kurz: Ja – ein Fork von Eclipse Theia und der Aufbau einer eigenen kommerziellen IDE sind rechtlich zulässig.**

Eclipse Theia steht unter **EPL v2** (Eclipse Public License 2.0), teils mit **GPL v2** als Secondary License. Die EPL v2 erlaubt ausdrücklich:

- **Kommerzielle Nutzung** – Nutzen, Ändern, Verteilen und **verkaufen** von Theia und darauf basierenden Produkten.
- **Derivative Works** – Ein Fork bzw. ein darauf aufbauendes Produkt (z. B. „Dedekind Studio“) ist ein zulässiges Derivative Work.

**Auflagen (weak copyleft):**

- **Theia-Code und eure Änderungen daran** müssen unter EPL v2 stehen und beim **Vertrieb** des Produkts **Quellcode** der (geänderten) Theia-Teile bereitgestellt werden.
- **Euer eigener, neuer Code** (z. B. Dedekind-spezifische Panels, Postgres-Anbindung, Branding), der nur per Schnittstelle (Extension, Link) mit Theia verbunden ist, kann **proprietär** bleiben – die EPL verlangt keine Weitergabe des gesamten Produkts unter EPL.
- Lizenzhinweise und Angabe der Änderungen sind erforderlich.

**Praktisch:** Ihr könnt Theia forken, als Basis für „Dedekind Studio“ nutzen und die IDE kommerziell vertreiben. Den Theia-Anteil (inkl. eurer Patches) müsst ihr unter EPL quelloffen machen; eure eigenen Module/Extensions könnt ihr proprietär lizenzieren. Für verbindliche Auslegung empfiehlt sich eine kurze rechtliche Prüfung (z. B. Lizenz-Compliance).

**Repo-Struktur:** Ja – dafür braucht ihr in der Regel **mehrere Repos** (oder zumindest eine klare Trennung):

| Repo | Inhalt | Lizenz | Sichtbarkeit |
|------|--------|--------|----------------|
| **Theia-Fork (EPL)** | Theia-Code + eure Änderungen daran (Branding, kleine Anpassungen). | EPL v2 | **Öffentlich** – Quellcode beim Vertrieb bereitstellen (z. B. GitHub/GitLab public). |
| **Dedekind Studio (proprietär)** | Eure eigenen Module: Dedekind-Integration, Einheiten-Panels, Postgres, KI, LaTeX-Export, Installer, etc. | Proprietär | **Privat** (oder nur Binär-Release ohne Source). |

Build/Pipeline baut dann aus beiden Quellen das auslieferbare Produkt (z. B. Electron-App). Alternative: ein **Monorepo** mit getrennten Verzeichnissen und Lizenzen – dann müsst ihr beim Vertrieb trotzdem den **Quellcode der EPL-Teile** bereitstellen (z. B. als Export oder öffentliches Subtree-Repo). Zwei Repos (ein öffentlicher Theia-Fork, ein privates Studio-Add-on) sind einfacher für Lizenz-Compliance und saubere Grenze zwischen EPL und proprietär.

### 2.4 Permissive Lizenzen: Fork ohne Quellcode-Veröffentlichung (wenig Verwaltungsaufwand)

Wenn ihr **keinen Code veröffentlichen** und **wenig Verwaltungsaufwand** wollt, sind IDEs/Editoren mit **permissiver Lizenz** (MIT, BSD, Apache 2.0) geeignet. Dort gilt: Fork erlaubt, kommerziell nutzen erlaubt, **keine Pflicht**, eure Änderungen oder euer Produkt quelloffen zu machen – nur Lizenzhinweise beibehalten.

| Kandidat | Lizenz | Quellcode-Veröffentlichung nötig? | Repos | Aufwand |
|----------|--------|----------------------------------|-------|---------|
| **JupyterLab** | **BSD 3-Clause** | **Nein** | Ein Repo reicht (Fork + eure Änderungen privat möglich) | Gering – Lizenz- und Copyright-Hinweise in Distribution beilegen. |
| **Spyder** | **MIT** | **Nein** | Ein Repo reicht | Gering – gleiche Auflage wie BSD. |
| **Monaco Editor** / **CodeMirror 6** | **MIT** | **Nein** | Kein „IDE-Fork“, nur Editor einbetten – eure App ist 100 % euer Code + MIT-Bibliothek(en). | Minimal – nur Attributions in App/Lizenzdatei. |
| **VS Code (Quellcode)** | **MIT** | **Nein** (für den Quellcode) | Theoretisch ein Repo; praktisch sehr großes Repo, Microsoft-Branding/Telemetrie in Binärversion. Für „saubere“ Distribution: [VSCodium](https://github.com/VSCodium/vscodium)-Fork nutzen (ebenfalls permissiv). | Hoch – Repo-Größe und Wartung; Lizenz-Compliance selbst bei MIT einfach. |
| **Eclipse Theia** | **EPL v2** | **Ja** – (geänderte) Theia-Teile beim Vertrieb quelloffen | Zwei Repos empfohlen (siehe 2.3) | Höher – getrennte Repos, Source-Release der EPL-Teile. |

**Empfehlung für wenig Verwaltungsaufwand:**

- **Voll-IDE-Fork ohne Veröffentlichungspflicht:** **JupyterLab** (BSD) oder **Spyder** (MIT). Ein Repo, Fork + eure Erweiterungen; nichts müsst ihr quelloffen machen, nur Lizenztexte beilegen.
- **Noch weniger Aufwand:** **Kein IDE-Fork**, sondern **Monaco** oder **CodeMirror 6** (MIT) als Editor einbetten und den Rest (Panels, Run, Plots, Postgres, KI) selbst bauen. Dann habt ihr keine Fork-Pflege und keine Veröffentlichungspflicht – nur eine kleine MIT-Attribution für den Editor.

**RStudio** ist unter **AGPL** – würde bei Nutzung als Netzwerkdienst sogar strengere Offenlegungspflichten auslösen; für „kein Code veröffentlichen“ ungeeignet.

### 2.5 Theia-Fork: „Massiv geändert“ – trotzdem quelloffen?

**Ja.** Unter EPL v2 kommt es nicht darauf an, *wie viel* ihr ändert, sondern ob der Code *von dem EPL-Programm abgeleitet* ist. Ein Fork von Eclipse Theia ist ein „Derivative Work“; auch **massive Änderungen** an diesem Fork ändern daran nichts – der Code bleibt abgeleitet von Theia.

- **Konsequenz:** Alle (geänderten) Theia-Anteile müsst ihr beim **Vertrieb** des Produkts unter EPL **quelloffen** bereitstellen. Ob ihr 10 % oder 90 % davon umgeschrieben habt, ist für die Lizenz unerheblich.
- **Proprietär bleiben können** nur Teile, die *keine* Ableitung von Theia sind – z. B. eigenständige Extensions/Module, die nur per API/Schnittstelle mit Theia kommunizieren und keinen Theia-Code enthalten. Sobald Theia-Code (oder davon abgeleiteter Code) in eurem Repo liegt, unterliegt dieser Teil der EPL.

Wer **keine** Quellcode-Veröffentlichung will, sollte eine **permissive** Basis wählen (z. B. JupyterLab/Spyder/Monaco wie in 2.4) statt Theia.

### 2.6 Performanz: Unterscheiden sich moderne IDEs? („Kompilieren sie alle gleich schnell?“)

**Kompilieren:** Wenn ihr in der IDE auf „Ausführen“ oder „Kompilieren“ klickt, startet die IDE in der Regel nur den **Compiler** (z. B. Dedekind-Compiler, gcc, rustc). Die **Kompiliergeschwindigkeit** hängt vom **Compiler** ab, nicht von der IDE. Alle IDEs, die denselben Compiler aufrufen, sind in dieser Hinsicht gleich „schnell“ – die IDE kompiliert nicht selbst.

**IDE-Performanz (Start, Tippen, Speicher):** Hier **unterscheiden** sich IDEs sehr wohl:

| Aspekt | Unterschied |
|--------|-------------|
| **Startzeit** | Electron-IDEs (VS Code, Theia, Cursor) starten oft langsamer als native Apps (z. B. Qt); durch Bundling und Code-Splitting lässt sich viel verbessern. |
| **Speicher** | Electron bringt Chromium + Node mit → höherer RAM-Bedarf; native oder schlanke Web-Editoren (z. B. CodeMirror-only) sind oft sparsamer. |
| **Editor-Reaktionszeit** | Abhängig von Editor-Engine (Monaco vs. CodeMirror), Größe des Dokuments, Syntax-Highlighting; große Bundles können erstes Laden verzögern. |
| **Sprach-Analyse (LSP)** | Auto-Completion, „Go to Definition“ etc. laufen im **Language Server** – die Geschwindigkeit hängt vom **LSP** ab (z. B. rust-analyzer vs. alter RLS), nicht primär von der IDE. |

**Fazit:** Beim **Kompilieren/Ausführen** eures Codes gibt es keinen IDE-Unterschied – der Compiler entscheidet. Beim **Gefühl** (Start, Flüssigkeit, Speicherverbrauch) schon – Electron vs. native, Editor-Größe und LSP-Qualität spielen eine Rolle.

**Nativer Editor – bestes Gefühl, auch auf älteren PCs:** Ja. Ein **nativer** Editor (z. B. Qt, GTK, oder plattformspezifisch Windows/macOS/Linux) bringt in der Regel:

- **Schnelleren Start** – kein Chromium/Node-Boot, direkte System-APIs.
- **Weniger RAM** – keine eingebettete Browser-Engine (Electron bringt oft 100–200+ MB Grundlast).
- **Flüssigeres Gefühl** – direkte Eingabe- und Rendering-Pipeline, kein JavaScript-Bridge.
- **Bessere Laufleistung auf älteren PCs** – geringere Mindestanforderungen (CPU, RAM), weniger Hintergrundlast.

Trade-off: **Entwicklungsaufwand** – native UIs pro Plattform oder mit Qt/GTK cross-platform bedeuten mehr Implementierungs- und Wartungsarbeit als eine Electron/Web-basierte IDE. Für „maximales Gefühl und breite Hardware-Unterstützung“ ist native die beste Wahl; für „schnell bauen, eine Codebasis für alle Plattformen“ bleibt Web/Electron oft pragmatischer.

### 2.7 Windows-First: Welche IDE oder welcher Editor als Fork?

Wenn ihr **zuerst auf Windows** starten wollt, kommen folgende Kandidaten in Frage:

| Kandidat | Lizenz | Art | Windows | Quellcode offenlegen? | Anmerkung |
|----------|--------|-----|---------|------------------------|-----------|
| **Lite XL** | **MIT** | Leichter nativer Editor (C + Lua) | Ja, Erstklassig | **Nein** | Sehr klein (<5 MB Bundle), Lua-Erweiterungen, nativer Look. Ideal für **native Basis + wenig Aufwand + keine Veröffentlichungspflicht**. |
| **Spyder** | **MIT** | Wissenschafts-IDE (Python, Qt) | Ja (Qt) | **Nein** | Läuft auf Windows mit Qt; Plots, Variablen-Explorer, Plugin-Architektur. Gute Passform für „Dedekind Studio“-Konzept, kein nativer Win32-Code. |
| **VSCodium** / **VS Code** | **MIT** | Electron-IDE | Ja | **Nein** | Kein nativer Editor; großer Repo. Wenn ihr „VS-Code-Feeling“ auf Windows wollt und Electron akzeptiert. |
| **Eclipse Theia** | **EPL v2** | Electron-IDE (TypeScript) | Ja | **Ja** (Theia-Anteile) | Wie zuvor: zwei Repos, Source-Release der Theia-Teile. |
| **Notepad++** | **GPL v3** | Nativer Editor (C++, Scintilla) | Ja, Windows-nativ | **Ja** (Copyleft) | Klassischer Windows-Editor; Fork würde unter GPL quelloffen bleiben. |
| **Scintilla** (nur Komponente) | **BSD-artig** | Nur Editor-Kern, keine IDE | Einbettbar in Windows-App | **Nein** | Kein voller IDE-Fork – Scintilla als Editor einbetten (z. B. C++/Qt oder Win32), Rest selbst bauen. Sehr leicht, native. |
| **Zed** | **GPL/AGPL** | Native Editor (Rust, GPUI) | Community (zed-win) | **Ja** (Copyleft) | Windows nicht Hauptziel der Zed-Entwicklung; GPUI ist Apache 2.0, Editor-Code GPL. |

**Empfehlung Windows-First:**

- **Native + wenig Verwaltung + kein Code veröffentlichen:** **Lite XL** (MIT) forken – kleiner Codebase, C/Lua, Windows-Support, Erweiterungen in Lua; IDE-Logik (Run, Plots, Postgres, KI) ergänzen.
- **Wissenschafts-IDE-Feeling + Windows:** **Spyder** (MIT) forken – Qt läuft gut auf Windows, Plots/Environment schon da, Dedekind als Backend/Kernel anbinden.
- **Nur Editor-Kern, Rest selbst:** **Scintilla** (BSD) in eine eigene Windows-App (z. B. C++/Qt oder Win32) einbetten – kein IDE-Fork, maximale Kontrolle, native, keine Veröffentlichungspflicht.

### 2.8 Dedekind Studio: Spyder-Fork mit nativ Python und Dedekind (Entscheidung)

**Dedekind Studio** wird als **Spyder-Fork** umgesetzt und bietet **nativ Python und Dedekind** in einer IDE:

- **Python** – wie in Spyder (IPython Console, Variable Explorer, Plots, Editor).
- **Dedekind** – Dedekind-Konsole über den **Dedekind Jupyter Kernel** (siehe `dedekind_jupyter_kernel/` in diesem Repo); optional .ddk-Editor mit Run und Syntax-Highlighting.

Konkrete Phasen, technische Schritte und Abhängigkeiten sind in **[Dedekind_Studio_Spyder_Fork.md](Dedekind_Studio_Spyder_Fork.md)** beschrieben. Der Dedekind-Kernel ist implementiert; nächster Schritt ist die Nutzung in Spyder („Connect to existing kernel“ oder Kernel-Auswahl bei neuer Konsole), danach der eigentliche Fork mit Branding „Dedekind Studio“.

---

## 3. Zwei Säulen im Überblick

| Säule | Ziel | Zielgruppe | Monetarisierung |
|-------|------|------------|-----------------|
| **Integration in bestehende IDEs** | Dedekind überall nutzbar; mehr Nutzer, mehr Sichtbarkeit. | Alle, die VS Code/Cursor/Jupyter nutzen. | Indirekt: mehr Adoption → mehr Abos für Dedekind Studio. |
| **Dedekind Studio** | Premium-IDE für Wissenschaftler: Einheiten, Plots, Postgres, LaTeX, lokale KI. | Forscher, Labore, Datenwissenschaftler (Datenschutz/Kosten-sensibel). | **Kommerziell**: Abo oder Lizenz; Finanzierung des Projekts. |

---

## 4. Phase 1: Dedekind in bestehenden IDEs (Reichweite)

**Ziel**: Dedekind in VS Code, Cursor und Jupyter nahtlos nutzbar machen; keine eigene IDE nötig, um Dedekind zu lernen oder einzusetzen.

### 4.1 VS Code / Cursor Extension

| Schritt | Inhalt | Aufwand (grober Richtwert) |
|---------|--------|----------------------------|
| **Sprach-Support** | Syntax-Highlighting für `.ddk`; ggf. TextMate-Grammatik aus bestehendem Lexer. | 1–2 Wochen |
| **Run & Debug** | „Dedekind ausführen“ über Compiler-Aufruf; Ausgabe in integrierter Konsole. Optional: Debug-Adapter (Breakpoints, Variablen). | 2–4 Wochen |
| **Einheiten-Hinweise** | LSP-ähnlich oder Inlay-Hints: angezeigte Einheit von Ausdrücken (z. B. `1[m] + 2[m]` → `3[m]`). Nutzung des bestehenden `units_checker` / Compiler. | 2–3 Wochen |
| **Fehlermeldungen** | Compiler-Fehler (inkl. Einheiten) in Problems-Liste und im Editor mit Zeile/Kontext. | 1 Woche |

**Technik**: Extension mit TypeScript/JavaScript; Aufruf des Dedekind-Compilers (CLI oder Python-API); optional Language Server Protocol (LSP) für erweiterte Features.

**Erfolgskriterium**: Nutzer können in VS Code/Cursor eine `.ddk`-Datei öffnen, ausführen und Einheiten-/Compiler-Fehler direkt im Editor sehen.

### 4.2 Jupyter Kernel für Dedekind

| Schritt | Inhalt | Aufwand |
|---------|--------|---------|
| **Kernel-Spec** | Jupyter-Kernel, der Dedekind-Code an den Compiler übergibt und Ergebnis/Plots zurückgibt. | 1–2 Wochen |
| **Zellen-Ausführen** | Code-Zelle → Compiler → Ausführung → stdout/stderr/Plots in Notebook. | 1 Woche |
| **Variablen/State** | Persistenter Laufzeit-Kontext über Zellen hinweg (wie bei Python-Kernel). | 1–2 Wochen |

**Nutzen**: Wissenschaftler arbeiten oft in Notebooks; Dedekind in Jupyter ermöglicht interaktives Experimentieren (Kinetik, Fitting, Plots) in bekannter Umgebung.

**Erfolgskriterium**: Jupyter-Notebook mit Dedekind-Kernel läuft; Zellen ausführbar, Plots anzeigbar.

### 4.3 Abhängigkeiten und Reihenfolge

- **Compiler-CLI stabil**: `python -m src.compiler.compiler <file>` mit klaren Exit-Codes und Fehlerausgabe (bereits weitgehend vorhanden).
- **Optional**: Compiler-API (z. B. `compile_source`, `run`) für Extension/Kernel ohne Subprocess.
- Empfehlung: **Zuerst VS Code Extension** (größte Sichtbarkeit), **danach Jupyter Kernel** (starker Nutzen für Wissenschaftler).

---

## 5. Phase 2: Dedekind Studio — Überlegene Wissenschaftler-IDE

**Ziel**: Dedekind Studio wird die **Referenz-IDE für Dedekind** mit Features, die in allgemeinen IDEs nicht oder nur mit Aufwand verfügbar sind. Orientierung an RStudio (domänenspezifisch, wissenschaftsnah), aber mit Fokus auf Einheiten, Datenbanken, LaTeX und **lokale KI**.

### 5.1 Feature-Übersicht (Zielbild)

| Feature | Beschreibung | Priorität |
|---------|--------------|-----------|
| **Einheitenchecks im Editor** | Live oder on-save: Einheiten-Fehler (z. B. `1[m] + 1[s]`) markieren, Inlay-Hints für Einheiten von Ausdrücken. Nutzung `units_checker` + Compiler. | Hoch |
| **Plots integriert** | Ausgabe von `plot()` und ggf. weiteren Visualisierungen in einem IDE-Panel (wie bereits in Dedekind Studio); Export als Bild/PDF. | Hoch (teilweise vorhanden) |
| **Native Postgres-Einbindung** | Verbindung zu PostgreSQL: Tabellen anzeigen, Abfragen in Dedekind oder SQL ausführen, Ergebnisse als Tensoren/Listen nutzen. Für Labor-Daten, Messreihen, Metadaten. | Hoch |
| **LaTeX-Export** | Auswahl von Code/Formeln → Export als LaTeX (bestehendes `export_to_latex`); in IDE als „Kopieren als LaTeX“ oder Sidebar-Vorschau. | Mittel |
| **Agentische KI (lokal)** | KI-Assistent ähnlich Cursor (Code vervollständigen, Erklären, Refaktorieren, „Dedekind-Code aus Beschreibung erzeugen“), aber **auf starken lokal ausgeführten Modellen** (z. B. Ollama, LM Studio, ggml). Begründung: **Datenschutz** (kein Code in die Cloud), **Kosten** (keine API-Gebühren). | Hoch |
| **Workspace / Environment** | Variablen, Konstanten, Einheiten nach Ausführung anzeigen (ähnlich RStudio Environment). | Mittel |
| **Quarto/Markdown** | Dedekind-Code-Blöcke in Markdown/Quarto-Dokumenten ausführen (optional, später). | Niedrig |

### 5.2 Agentische KI mit lokalen Modellen — Detail

- **Anforderung**: Kein zwingender Upload von Code zu externen Diensten; Nutzer mit sensiblen Daten (Pharma, Klinik, Behörden) können KI nutzen.
- **Technik**: Integration mit **Ollama**, **LM Studio** oder vergleichbaren lokalen LLM-Servern; IDE sendet nur an localhost. Modelle: z. B. Llama, Mistral, Qwen (Code-fähig).
- **Funktionen**: Code-Vervollständigung, „Erkläre diesen Block“, „Schreibe Dedekind-Code für …“, Refactoring, Einheiten-Erklärungen. Optional: Hybrid (lokal Standard, Cloud optional für Nutzer, die es wünschen).
- **Abgrenzung zu Cursor**: Cursor setzt auf Cloud-Modelle; Dedekind Studio positioniert sich mit **„KI ohne Daten zu verlassen“** und **kalkulierbaren Kosten** für Teams.

### 5.3 Native Postgres-Einbindung — Detail

- **Anwendung**: Labor-Daten, Messreihen, Stichproben-Metadaten in Postgres; Abfragen in der IDE, Ergebnis direkt in Dedekind weiterverarbeiten (z. B. `fit`, `plot`).
- **Umsetzung**: In Dedekind Studio ein **Daten-Panel** oder **Query-Editor**: Verbindung (Host, Port, DB, User, Passwort) konfigurieren; SQL ausführen oder einfache Dedekind-API (z. B. `db_query("SELECT ...")`) die Resultate als Listen/Tensoren bereitstellt. Optional: Dedekind-Sprache um `sql`-Blöcke oder eine `postgres`-Bibliothek erweitern (Runtime).
- **Lizenz**: PostgreSQL ist Open Source; Treiber (z. B. psycopg2, node-postgres) verfügbar.

### 5.4 Implementierungs-Reihenfolge (Phase 2)

| Stufe | Inhalt | Abhängigkeit |
|-------|--------|---------------|
| **2a** | Einheitenchecks im Editor (Fehler markieren, Hover); Plots-Panel robuster und exportierbar. | Bestehender Compiler, `units_checker` |
| **2b** | LaTeX-Export aus IDE (Button/ Kontextmenü „Als LaTeX kopieren“). | `export_to_latex` |
| **2c** | Lokale KI-Integration (Ollama/LM Studio): Chat-Panel, Code-Actions (Erklären, Vervollständigen). | Lokaler LLM-Server, API-Anbindung |
| **2d** | Postgres: Verbindungsdialog, Query-Editor, Ergebnis-Tabelle; optional Dedekind-Runtime-Funktion `db_query`. | Postgres-Client in Studio-Backend |
| **2e** | Workspace/Environment-Panel; optional Quarto/Markdown. | Laufzeit-Status vom Compiler/Runtime |

---

## 6. Phase 3: Kommerzialisierung von Dedekind Studio

**Ziel**: Dedekind Studio ist das **kostenpflichtige Produkt**; Erlöse finanzieren Entwicklung von Compiler, Sprache, Doku und Community.

### 6.1 Lizenzmodell (Vorschlag)

| Stufe | Zielgruppe | Inhalt | Einnahme |
|-------|------------|--------|----------|
| **Kostenlos / Community** | Einsteiger, Open-Source-Projekte | Dedekind Studio mit eingeschränkten Features (z. B. keine Postgres, keine KI oder nur eingeschränkte lokale KI); oder zeitlich begrenzte Vollversion. | Sichtbarkeit, Adoption |
| **Pro / Team** | Einzelne Forscher, kleine Teams | Volle Features: Einheitenchecks, Plots, Postgres, LaTeX, lokale KI; Updates; optional Support. | Abo (monatlich/jährlich) |
| **Enterprise** | Institute, Firmen, Behörden | Pro-Features + zentrales Lizenzmanagement, SSO, Compliance, priorisierter Support. | Jahreslizenz, Volumen |

**Dedekind-Sprache und -Compiler** bleiben **Open Source** (MIT o. ä.); Nutzer können weiterhin in VS Code/Cursor/Jupyter ohne Dedekind Studio arbeiten. Dedekind Studio ist der **Premium-Arbeitsplatz** für alle, die maximale UX und integrierte Wissenschafts-Features wollen.

### 6.2 Finanzfluss

- **Einnahmen**: Dedekind Studio Abos/Lizenzen.
- **Verwendung**: Entwicklung Dedekind Studio (IDE, Postgres, KI-Integration); Entwicklung Dedekind-Compiler/Runtime (Open Source); Doku, Beispiele, Community-Events; ggf. Stipendien für Beiträgende.
- **Transparenz**: Optional „Open Book“ oder jährlicher Bericht, welcher Anteil in Open Source fließt — stärkt Vertrauen und Community.

### 6.3 Rechtliches und Abgrenzung

- **Datenschutz**: Lokale KI als Standard dokumentieren; keine Nutzerdaten/Code an Drittanbieter ohne explizite Einwilligung (bei optionaler Cloud-KI).
- **Nutzungsbedingungen**: Klarstellen: Dedekind Studio ist proprietäre Software; Dedekind Language/Compiler sind Open Source und unabhängig nutzbar.
- **Kompatibilität**: VS Code Extension und Jupyter Kernel bleiben **kostenlos**, damit die Sprache weit verbreitet bleibt; Dedekind Studio wirbt mit Mehrwert (Postgres, KI, integrierte Wissenschafts-Features).

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
| **Phase 2a–2c** | Dedekind Studio: Einheiten im Editor, Plots, LaTeX, lokale KI. | 3–6 Monate |
| **Phase 2d–2e** | Postgres, Workspace-Panel; optional Quarto. | 2–4 Monate |
| **Phase 3** | Lizenzmodell, Bezahlfluss, Pro/Enterprise-Tiers; Launch von Dedekind Studio kommerziell. | Parallel zu Phase 2, Launch nach 2a–2c |

---

## 9. Zusammenfassung

- **Dedekind in bestehenden IDEs** (VS Code, Cursor, Jupyter) erhöht Reichweite und Beliebtheit; **Dedekind Studio** wird die **Premium-IDE für Wissenschaftler** mit Einheitenchecks, Plots, Postgres, LaTeX und **agentischer KI auf lokalen Modellen**.
- **Dedekind Studio ist kommerziell**; Erlöse finanzieren das Gesamtprojekt; **Sprache und Compiler bleiben Open Source**.
- Diese Roadmap ergänzt die [Commercialization_Options](Commercialization_Options.md) um die konkrete IDE- und Produktstrategie und sollte mit der [Features Implementation Roadmap](Features_Implementation_Roadmap.md) sowie der Chemie/Biologie-Roadmap abgestimmt bleiben (Compiler-Features, Einheiten, LaTeX).
