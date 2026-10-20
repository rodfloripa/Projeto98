# Comparação de Métricas para Identificar a Complexidade dos Textos

# 1. Objetivo

<p align="justify">Este projeto compara medidas automáticas aplicadas a seis frases em inglês, previamente classificadas em três categorias: básico, médio e avançado. O código calcula a perplexidade de um modelo de linguagem, o índice Flesch-Kincaid, a Facilidade de Leitura de Flesch e um índice híbrido que combina a perplexidade com o Flesch-Kincaid. Também registra a quantidade de palavras e sílabas de cada frase.</p>

<p align="justify">O objetivo é observar qual medida acompanha melhor as categorias atribuídas aos textos. Os resultados mostram que as métricas não medem exatamente a mesma coisa: algumas estimam a dificuldade de leitura com base na estrutura das frases, enquanto a perplexidade estima o quanto o texto é previsível para um modelo de linguagem.</p>

# 2. Dados Utilizados

<p align="justify">O conjunto possui seis frases: duas classificadas como básicas, duas como médias e duas como avançadas. Essa quantidade é útil para uma demonstração inicial, mas é pequena para sustentar conclusões gerais. Além disso, as categorias de referência foram atribuídas previamente; elas não são descobertas automaticamente pelas métricas.</p>

# 3. Métricas Calculadas

<p align="justify"><b>Perplexidade:</b> é obtida a partir da perda calculada pelo modelo <code>distilgpt2</code>, aplicando a função exponencial. Valores maiores indicam que o texto é menos previsível para esse modelo. Isso não significa necessariamente que o texto seja mais difícil para uma pessoa, pois a previsibilidade também depende do vocabulário e dos padrões aprendidos pelo modelo.</p>

<p align="justify"><b>Flesch-Kincaid:</b> estima o nível escolar de leitura a partir de características estruturais, incluindo o comprimento das frases e das palavras. Em geral, valores maiores indicam maior exigência de leitura. Por isso, essa medida tem uma relação direta com a tentativa de distinguir textos básicos, médios e avançados.</p>

<p align="justify"><b>Facilidade de Leitura de Flesch:</b> estima a facilidade de leitura. Valores altos indicam textos mais fáceis; valores baixos ou negativos indicam textos mais difíceis. A direção dessa escala é oposta à do Flesch-Kincaid: quando a dificuldade aumenta, o nível escolar estimado tende a subir, enquanto a facilidade de leitura tende a cair.</p>

<p align="justify"><b>Índice híbrido:</b> o código normaliza a perplexidade e o Flesch-Kincaid para uma escala de 0 a 1 e combina os resultados, atribuindo peso de 60% ao Flesch-Kincaid e 40% à perplexidade. O resultado é multiplicado por 100. Esse índice depende dos valores mínimo e máximo do próprio conjunto; portanto, não representa uma escala universal de complexidade.</p>

# 4. Partes Mais Importantes do Código

<p align="justify">A primeira etapa define as frases, seus identificadores e as categorias de referência. Essas categorias permitem comparar as medidas calculadas com os níveis esperados.</p>

```python
novas_frases = [
    {
        "doc_id": 1,
        "nivel": "Nível 1 (Básico)",
        "texto": "The small dog, owned by Zachary, barked loudly at the blue mailbox."
    },
    {
        "doc_id": 2,
        "nivel": "Nível 2 (Médio)",
        "texto": "The company expects to increase its quarterly revenue by implementing new strategic plans developed during the annual board meetings."
    }
]
```

<p align="justify">O trecho apresenta dois exemplos da estrutura usada no caderno. O conjunto completo contém seis frases, cada uma com uma categoria previamente atribuída.</p>

<p align="justify">Na etapa seguinte, o modelo calcula a perda associada aos tokens do texto. A exponencial dessa perda fornece a perplexidade.</p>

```python
inputs = tokenizer(texto, return_tensors="pt")
input_ids = inputs["input_ids"]
outputs = model(input_ids, labels=input_ids)

loss_global = outputs.loss.item()
perplexidade = round(np.exp(loss_global), 2)
```

<p align="justify">O código calcula também as medidas tradicionais de legibilidade. A contagem de palavras e sílabas ajuda a interpretar como o tamanho e a estrutura das palavras influenciam os índices.</p>

```python
fk_grade = round(textstat.flesch_kincaid_grade(texto), 2)
reading_ease = round(textstat.flesch_reading_ease(texto), 2)
syllable_count = textstat.syllable_count(texto)
lexicon_count = textstat.lexicon_count(texto)
```

<p align="justify">Por fim, o código normaliza as duas medidas escolhidas e cria o índice híbrido. A normalização coloca os valores em uma escala comum, mas não garante que a combinação classifique corretamente os níveis de referência.</p>

```python
def min_max_scale(arr):
    denom = arr.max() - arr.min()
    if denom == 0:
        return np.zeros_like(arr)
    return (arr - arr.min()) / denom

ppl_norm = min_max_scale(ppl_vals)
fk_norm = min_max_scale(fk_vals)

peso_flesch = 0.6
peso_ppl = 0.4
ind_hibrido = (fk_norm * peso_flesch) + (ppl_norm * peso_ppl)
ind_hibrido_100 = np.round(ind_hibrido * 100, 2)
```

# 5. Resultados Obtidos

<p align="justify">A execução registrada no caderno produziu os seguintes valores. A tabela está organizada por índice híbrido crescente, conforme a ordenação realizada no código.</p>

| Identificador | Categoria de referência | Perplexidade | Flesch-Kincaid | Facilidade de Leitura | Índice híbrido |
|---:|---|---:|---:|---:|---:|
| 1 | Básico | 113,45 | 4,82 | 81,86 | 17,98 |
| 3 | Básico | 185,64 | 5,21 | 84,45 | 41,32 |
| 2 | Médio | 104,39 | 15,42 | 18,35 | 51,17 |
| 5 | Médio | 127,10 | 19,43 | -21,11 | 71,70 |
| 4 | Avançado | 54,48 | 22,51 | -39,63 | 60,00 |
| 6 | Avançado | 95,40 | 19,37 | -18,88 | 61,83 |

<p align="justify">Os resultados do Flesch-Kincaid seguem, em geral, a ordem esperada: as duas frases básicas têm valores baixos; as frases médias e avançadas apresentam valores mais altos. Há, porém, uma sobreposição: a frase 5, classificada como média, tem valor 19,43, muito próximo ou acima do valor observado em uma das frases avançadas.</p>

<p align="justify">A Facilidade de Leitura de Flesch apresenta um padrão semelhante, mas em sentido inverso. As duas frases básicas têm valores altos, enquanto as frases avançadas apresentam valores negativos. A frase 5, de nível médio, também apresenta facilidade de leitura negativa, aproximando-se dos textos avançados.</p>

<p align="justify">A perplexidade não separa bem as categorias neste conjunto. Por exemplo, uma frase básica tem perplexidade 185,64, maior do que as duas frases avançadas, com valores 54,48 e 95,40. Isso mostra que a imprevisibilidade para o modelo não corresponde necessariamente à dificuldade de leitura humana.</p>

<p align="justify">O índice híbrido reúne duas perspectivas complementares: a dificuldade estrutural estimada pelo Flesch-Kincaid e a imprevisibilidade medida pela perplexidade. Embora a tabela apresente algumas inversões — como a frase média com índice 71,70 —, a combinação oferece uma visão mais abrangente do que cada medida isolada, desde que seus resultados sejam interpretados em conjunto com as categorias de referência.</p>

# 6. Conclusão

<p align="justify"><b>Neste projeto, o índice híbrido é considerado a melhor abordagem geral</b> porque combina duas informações diferentes: o nível escolar estimado pelo Flesch-Kincaid e a previsibilidade do texto para o modelo, medida pela perplexidade. Essa combinação permite analisar a complexidade sob mais de uma perspectiva, em vez de depender exclusivamente de uma única característica textual.</p>

<p align="justify">Os resultados mostram que o Flesch-Kincaid e a Facilidade de Leitura de Flesch acompanham de maneira útil a dificuldade estrutural das frases. A perplexidade, por sua vez, mede o quanto o texto é inesperado para o modelo <code>distilgpt2</code>, e não diretamente a dificuldade para uma pessoa. O índice híbrido integra essas informações e, por isso, é a opção mais completa para a análise proposta.</p>

<p align="justify">É importante observar que o conjunto contém apenas seis frases e que o índice híbrido depende dos pesos escolhidos — 60% para o Flesch-Kincaid e 40% para a perplexidade — e da normalização feita sobre esse próprio conjunto. Assim, considerá-lo a melhor abordagem geral neste projeto não significa que já tenha sido comprovado como o classificador de maior taxa de acerto. Para confirmar seu desempenho, recomenda-se ampliar a quantidade de textos e avaliar as métricas em dados adicionais.</p>
