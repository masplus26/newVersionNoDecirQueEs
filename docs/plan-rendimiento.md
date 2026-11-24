# Plan: rendimiento y compatibilidad de GPU

Objetivo: frame de 33 ms (30 FPS) en escenas sintéticas propias que imiten una
carga de mundo abierto. Ninguna fase usa datos de un juego: se valida con
programas D3D12 y shaders propios. Que un título real llegue a 30 FPS **no
queda demostrado** por este plan.

## Datos medidos (2026-11-24)

| | RTX 5070 Ti Laptop | Radeon 610M |
|---|---|---|
| Wave (lanes) | 32 solamente | 32 a 64 |
| Mesh shaders | sí (tier 1) | **no** |
| Atómicas de 64 bits en recursos tipados | sí | sí |
| Depth bounds | sí | sí |
| GPU upload heaps (VRAM escribible por CPU) | sí | sí |
| Ray tracing | tier 1.2 | tier 1.1 |

Copiar 174 MB de RAM a RAM cuesta ~16 ms y más hilos no ayudan: hay que mover
menos bytes, no moverlos más rápido.

## Hecho

- El emulador usa la GPU de alto rendimiento. La 610M no soporta mesh shaders.
- Escrituras de shaders comparadas una vez por envío: 454 → ~33 ms (sintético).
- Fase 0: `cargo gpu-bench` (crates/victory-gpu/src/bench.rs). Repetible a
  ±2,7 % entre procesos; imagen y memoria del título iguales a WARP.
- Escritura de vuelta al título sin `WriteProcessMemory`: frame del banco de
  764–858 a 242–287 ms (perfil de abajo).
- Descartado por medición: copias multihilo y LTO (sin mejora).

## Perfil del banco (RTX 5070 Ti, 6 envíos de 24 vistas sobre 174 MiB)

| Parte del frame | ms |
|---|---|
| Preparar: snapshots y resto | ~87 |
| Preparar: texturas | ~10 |
| Preparar: copias de buffers | ~2,5 |
| GPU, con su espera | ~20 |
| Copia de vuelta (compara 6 × 174 MB, une y escribe 57 MB) | ~129 |
| **Total** | **~252** |

La GPU no es el límite del banco: el snapshot y la comparación son ~85 %. La
fase 1 es la que más rinde; la 2 (VRAM) aporta como mucho ~20 ms acá.
Es una carga inventada: las proporciones de un título real pueden ser otras.

## Fase 0: banco de pruebas (hecha)

- Escena sintética: muchos envíos, heap de 174 MB, vistas solapadas, dispatch
  que escriben, texturas y dibujos indirectos. Referencia de imagen en WARP.
- Tiempos por frame: preparar, GPU, copiar de vuelta, bytes movidos.
- **Listo cuando:** un comando da un perfil repetible (±5 %).
- Riesgo bajo. El perfil puede reordenar las fases 1 a 4.

## Estado de la fase 1 (hecha)

Buffers del título de 4 MiB o más en VRAM (`vram.rs`), comparados en la GPU;
solo viajan las piezas de 64 KiB que cambiaron. Frame del banco: ~252 → ~70 ms
(preparar ~14, GPU ~14, copia de vuelta ~39). Adelanta parte de la fase 2.
La meta de "menos de 3 ms" por envío del caso de 24 vistas no se midió así;
en el banco, con ~9 MB cambiados por envío, la copia de vuelta cuesta ~6,5 ms.

Texturas: se reconvierten solo las páginas de 64 KiB cambiadas (vigiladas o con
hash por página): ~10 → ~3 ms. Frame del banco: ~55 ms. Lo que queda es sobre
todo el hash de las no vigiladas. La subida a la textura de la GPU sigue
siendo completa.

Pendiente
correr con el debug layer de D3D12 (no instalado en este equipo).

## Fase 1: menos trabajo de CPU por envío

1. Escrituras de shaders: copia y comparación en GPU contra una copia previa,
   máscara por página de 64 KiB; la CPU lee solo páginas cambiadas.
2. Hash de texturas: solo si la página cambió; hash por bloques.
- **Listo cuando:** el caso de 24 vistas baja de ~33 ms a menos de 3 ms, y los
  tests de `shadow` siguen pasando (los datos del título nunca se pisan).
- Riesgo medio: el costo de la comparación en GPU compite con el render.

## Fase 2: buffers en VRAM

1. Solo lectura para shaders: GPU upload heap, con caída a RAM si no hay.
2. Escritos por shaders: heap por defecto más readback de las páginas sucias
   (depende de la fase 1). La memoria write-combined no se lee desde la CPU.
3. Presupuesto de VRAM (12 GB) con expulsión.
- **Listo cuando:** la parte de GPU del banco mejora y ninguna lectura de CPU
  toca VRAM. Si no mejora, se revierte.
- Riesgo medio.

## Fase 3: wave64 sobre wave32

Afecta a los shaders de cómputo que fijan `WaveSize(64)` (`lower.rs`,
`console_lanes`). Los que no lo necesitan ya corren.

1. Contar con `victory translate` qué operaciones cruzan lanes.
2. Grupos de hasta 64 hilos: emular con groupshared y barreras (`mbcnt`,
   `readlane`, `writelane`, permutaciones, EXEC de 64 bits).
3. Grupos mayores y bucles dependientes de otros lanes: caso por caso.
- **Listo cuando:** los programas de prueba dan igual resultado en 610M (wave64
  real) y RTX (emulado).
- Riesgo alto: las barreras dentro de control de flujo divergente no son
  válidas; parte de los programas puede no ser emulable.

## Fase 4: funciones de GPU que faltan

Orden por costo: depth bounds → consultas reales → indirectos con argumentos
raíz → transiciones por subrecurso → atómicas de imagen de 64 bits → HDR y
escalado → geometría combinada → metadatos de render targets → ray tracing.

- **Listo cuando:** cada función tiene un test propio comparado con WARP o con
  un resultado calculado.
- Los shaders sin traducir se abordan por opcode, con el método de
  `emit/diff_tests.rs`.

## Fase 5: CPU y GPU en paralelo (al final)

Hoy `Host::execute` espera cada envío. Snapshots, rangos pendientes y avisos
de fin asumen uno a la vez.

1. Estado por envío en vez de por buffer (snapshots, escrituras).
2. Hilo de finalización: copia de vuelta y después avisos al título, en orden.
3. Hasta 2 o 3 envíos en vuelo; esperar solo si el siguiente lee lo escrito.
4. Interruptor para volver al modo serial.
- **Listo cuando:** el banco da la misma imagen en ambos modos, la prueba de
  estrés corre 1 hora sin errores del debug layer, y el tiempo por frame baja.
- Riesgo muy alto: corrupción o cuelgues intermitentes.

## Reglas

- Una rama y un PR por fase. Medir antes y después; sin mejora, se revierte.
- `cargo clean` al comparar versiones: compartir `target` entre copias del
  código reutiliza binarios viejos.
- Mantener el tope de iteraciones de shaders: evita colgar el driver.

## Estimación

En sesiones de trabajo conmigo, muy aproximada: fase 0: 1; fase 1: 2 a 3;
fase 2: 2 a 3; fase 3: 4 a 8; fase 4: 10 o más; fase 5: 5 a 10.
