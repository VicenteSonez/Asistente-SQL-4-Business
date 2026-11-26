# Guion del video del Deliverable 4 (máximo 3 minutos)

Requisito del enunciado: grabación de pantalla del sistema con el agente ajustado corriendo sobre
entradas reales, con los mensajes entre agentes visibles, y el agente base y el ajustado sobre la misma
entrada.

## Antes de grabar

- Terminal de PowerShell en la raíz del repositorio, en la rama `deliverable4`, con letra grande
  (Ctrl + rueda) y la ventana maximizada.
- Comando que se graba (H100 en Modal, ~5 min, ~US$0,5; la carga del modelo toma ~1 min y se corta en
  la edición):

```
modal run scripts/modal_d4.py::demo --ids q12,q11-p
```

- Pestañas abiertas para mostrar al final: el repo de Hugging Face
  (https://huggingface.co/vicentesonez/sql4business-planificador-qlora) y `docs/deliverable4.pdf`.

## Guion

| Tiempo | En pantalla | Qué decir |
|---|---|---|
| 0:00–0:20 | PDF: grafo de agentes | "SQL4Business responde preguntas de negocio con cuatro agentes. En el D3, los errores nacían en el Planificador: cuando el Verificador le devolvía el plan con el motivo exacto, repetía el mismo plan. En el D4 ajustamos solo el Planificador con QLoRA; los otros tres agentes usan el modelo base." |
| 0:20–0:40 | PDF: datos y curva de pérdida | "Entrenamos con 1.282 pares prompt → plan, construidos y verificados por código, sin preguntas de evaluación. 290 pasos en un H100, 17 minutos. El Verificador ahora revisa el resultado antes de que el Reportero redacte, por el comentario al D3." |
| 0:40–1:35 | Terminal: q12, Planificador **base** y luego **ajustado** | "q12: '¿el ingreso de agosto fue mayor o menor que el de marzo?'. El Planificador base pone agosto primero, el Verificador lo devuelve y el base repite el mismo plan: −37 %. Ahora el ajustado, con la misma entrada…" (comentar lo que muestra la corrida: orden de los pasos, mensaje del Verificador, resultado final). |
| 1:35–2:25 | Terminal: q11-p, base y ajustado | "q11-p es una paráfrasis externa que nunca vimos al diseñar: total y parte de una participación." (comentar la diferencia entre ambos). |
| 2:25–2:50 | PDF: tabla de resultados y caso de falla | Leer la fila (3) contra la (4) y el control. "Y un caso donde el ajustado todavía falla: …" (ver `README.md`, sección Caso de falla). |
| 2:50–3:00 | Hugging Face | "El adaptador está publicado; la evaluación descarga esta revisión exacta. Todo se reproduce con `scripts/modal_d4.py`." |

Al terminar, subir el video con acceso abierto (YouTube no listado o Drive con "cualquiera con el
enlace") y pegar el link en el README y en `\VideoURL` de `docs/deliverable4.tex`.
