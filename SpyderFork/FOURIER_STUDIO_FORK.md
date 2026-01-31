# Fourier Studio – Spyder-Fork

Dieser Ordner enthält einen **Fork von [Spyder](https://github.com/spyder-ide/spyder)** für **Fourier Studio**. Fourier Studio bietet nativ **Python** und **Fourier** in einer IDE (siehe [Fourier_Studio_Spyder_Fork.md](../Documentation/Fourier_Studio_Spyder_Fork.md) im Hauptrepo).

## Upstream

- **Original:** https://github.com/spyder-ide/spyder  
- **Lizenz:** MIT (unverändert)

## Nächste Schritte (Fourier Studio)

1. **Branding:** Name, Icons und Fenstertitel auf „Fourier Studio“ umstellen.
2. **Fourier-Kernel:** Sicherstellen, dass der Fourier Jupyter Kernel (`fourier_jupyter_kernel/` im Hauptrepo) beim Start einer neuen Konsole wählbar ist.
3. **Optional:** Standard-Kernel für neue Konsolen auf „Fourier“ setzbar machen; `.fourier`-Editor mit Run und Syntax-Highlighting.

## Upstream aktualisieren

```bash
cd SpyderFork
git fetch origin
git merge origin/master
```
