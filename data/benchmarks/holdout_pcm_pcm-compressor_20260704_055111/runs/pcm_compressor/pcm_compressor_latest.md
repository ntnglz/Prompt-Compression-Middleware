# Benchmark PCM — example_prompts.json

- **Generado:** 2026-07-04T05:50:49.814047+00:00
- **Modelo:** `pcm-compressor`
- **Estrategia:** `balanced`
- **Thinking:** `auto/off`
- **Similitud semántica (LLM):** sí

## Resumen

| Métrica | Valor |
|---------|-------|
| Prompts | 10 |
| Ratio medio | 21.02% |
| Ratio min / max | 0.00% / 52.94% |
| Tokens ahorrados | 68 |
| Similitud formato (media) | 60.00% |
| Similitud semántica (media) | 92.00% |
| Tiempo medio | 1969 ms |

## Detalle por prompt

| ID | Categoría | Ratio | Formato | Semántica | Original → Comprimido |
|----|-----------|-------|---------|-----------|------------------------|
| holdout_000 | holdout | 53% | 100% | 70% | Examina detenidamente el siguiente códig... → `TASK=review INPUT=typescript CHECK=logic,anti_patterns,perf FORMAT=list` |
| holdout_001 | holdout | 39% | 100% | 90% | Revisa este código javascript buscando b... → `TASK=review INPUT=javascript CHECK=race,leak,perf FORMAT=markdown ORDER=severity` |
| holdout_002 | holdout | 33% | 100% | 90% | Traduce este documento marketing del en ... → `TASK=translate FROM=en TO=es STYLE=formal DOMAIN=marketing` |
| holdout_003 | holdout | 0% | 0% | 100% | Create an executive summary of this annu... → `Create an executive summary of this annual financial report with key metrics and strategic highlights.` |
| holdout_004 | holdout | 0% | 0% | 100% | Explica qué es cybersecurity a un públic... → `Explica qué es cybersecurity a un público principiante usando analogías y ejemplos cotidianos.` |
| holdout_005 | holdout | 0% | 0% | 100% | Write a professional blog post introduct... → `Write a professional blog post introduction about cloud_migration for developers. Tone: informative and concise.` |
| holdout_006 | holdout | 38% | 100% | 90% | Comparar estos dos algoritmos de ordenam... → `TASK=compare ITEMS=quicksort,mergesort CRITERIA=time_complexity,space_complexity,use_cases` |
| holdout_007 | holdout | 0% | 0% | 100% | Create a detailed roadmap for building a... → `Create a detailed roadmap for building a SaaS dashboard with role-based access, analytics, and export features.` |
| holdout_008 | holdout | 20% | 100% | 90% | Revisa este contrato y señala cláusulas ... → `TASK=review INPUT=contract FOCUS=problematic_clauses CHECK=cancellation_terms,penalties` |
| holdout_009 | holdout | 27% | 100% | 90% | Diseña un esquema de base de datos para ... → `TASK=design SCHEMA=ecommerce ENTITIES=users,products,orders,reviews REQUIRE=relationships,indexes OPTIMIZE=performance` |

## Esperado vs obtenido

### holdout_000 (holdout)

**Esperado:** `TASK=review INPUT=typescript CHECK=logic,anti_patterns,perf FORMAT=list`

**Obtenido:** `TASK=review INPUT=typescript CHECK=logic,anti_patterns,perf FORMAT=list`


### holdout_001 (holdout)

**Esperado:** `TASK=review INPUT=javascript CHECK=race,leak,perf FORMAT=markdown ORDER=severity`

**Obtenido:** `TASK=review INPUT=javascript CHECK=race,leak,perf FORMAT=markdown ORDER=severity`


### holdout_002 (holdout)

**Esperado:** `TASK=translate FROM=en TO=es STYLE=formal DOMAIN=marketing`

**Obtenido:** `TASK=translate FROM=en TO=es STYLE=formal DOMAIN=marketing`


### holdout_003 (holdout)

**Esperado:** `TASK=summarize INPUT=annual_report FORMAT=list HIGHLIGHT=metrics,strategic_highlights`

**Obtenido:** `Create an executive summary of this annual financial report with key metrics and strategic highlights.`

- Claves faltantes: `FORMAT, HIGHLIGHT, INPUT, TASK`

### holdout_004 (holdout)

**Esperado:** `TASK=explain TOPIC=cybersecurity AUDIENCE=beginner STYLE=simple USE=analogies,examples`

**Obtenido:** `Explica qué es cybersecurity a un público principiante usando analogías y ejemplos cotidianos.`

- Claves faltantes: `AUDIENCE, STYLE, TASK, TOPIC, USE`

### holdout_005 (holdout)

**Esperado:** `TASK=write TYPE=blog TOPIC=cloud_migration AUDIENCE=developers STYLE=informative TONE=concise`

**Obtenido:** `Write a professional blog post introduction about cloud_migration for developers. Tone: informative and concise.`

- Claves faltantes: `AUDIENCE, STYLE, TASK, TONE, TOPIC, TYPE`

### holdout_006 (holdout)

**Esperado:** `TASK=compare ITEMS=quicksort,mergesort CRITERIA=time_complexity,space_complexity,use_cases`

**Obtenido:** `TASK=compare ITEMS=quicksort,mergesort CRITERIA=time_complexity,space_complexity,use_cases`


### holdout_007 (holdout)

**Esperado:** `TASK=create_plan TYPE=project DOMAIN=saas_dashboard FEATURES=rbac,analytics,export`

**Obtenido:** `Create a detailed roadmap for building a SaaS dashboard with role-based access, analytics, and export features.`

- Claves faltantes: `DOMAIN, FEATURES, TASK, TYPE`

### holdout_008 (holdout)

**Esperado:** `TASK=review INPUT=contract FOCUS=problematic_clauses CHECK=cancellation_terms,penalties`

**Obtenido:** `TASK=review INPUT=contract FOCUS=problematic_clauses CHECK=cancellation_terms,penalties`


### holdout_009 (holdout)

**Esperado:** `TASK=design SCHEMA=ecommerce ENTITIES=users,products,orders,reviews REQUIRE=relationships,indexes OPTIMIZE=performance`

**Obtenido:** `TASK=design SCHEMA=ecommerce ENTITIES=users,products,orders,reviews REQUIRE=relationships,indexes OPTIMIZE=performance`
