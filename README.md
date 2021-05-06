# row-column-intersection
This project makes available the code and data from our NAACL 2021 paper: "Capturing Row and Column Semantics in Transformer Based Question Answering over Tables"
It is available now on [arxiv](https://arxiv.org/abs/2104.08303)

The commands to process each dataset are under the datasets directory in README.md for each dataset:

* [WikiSQL](datasets/wikisql/README.md)
* [TabMCQ](datasets/tabmcq/README.md)
* [WikiTableQuestions](datasets/wtq/README.md)


# Overview

The structure is to convert each dataset (WikiSQL, WikiTableQuestions and TabMCQ) into a standard jsonl format, for example:
```json
{"id":"nt-95",
"table_id":"csv\/203-csv\/189.csv",
"question":"how long was the race in the all-africa games (distance)?",
"target_column":4,"answers":["10,000 m"],
"header":["Year","Competition",               "Venue",                     "Position","Notes"],
"rows":[["1994", "World Junior Championships","Lisbon, Portugal",          "2nd",     "5,000 m"],
        ["1995", "All-Africa Games",          "Harare, Zimbabwe",          "2nd",     "5,000 m"],
        ["1995", "All-Africa Games",          "Harare, Zimbabwe",          "2nd",     "10,000 m"],
        ["1997", "World Championships",       "Athens, Greece",            "7th",     "10,000 m"],
        ["1999", "All-Africa Games",          "Johannesburg, South Africa","3rd",     "10,000 m"],
        ["2001", "World Championships",       "Edmonton, Canada",          "9th",     "10,000 m"]]}
```

For both training and apply, these tables are decomposed into sequence-pair classification instances for rows and columns independently.

Columns:
```json
{"id":"nt-95:0","text_a":"how long was the race in the all-africa games (distance)?","text_b":"Year * 1994 * 1995 * 1995 * 1997 * 1999 * 2001","label":false}
{"id":"nt-95:1","text_a":"how long was the race in the all-africa games (distance)?","text_b":"Competition * World Junior Championships * All-Africa Games * All-Africa Games * World Championships * All-Africa Games * World Championships","label":false}
{"id":"nt-95:2","text_a":"how long was the race in the all-africa games (distance)?","text_b":"Venue * Lisbon, Portugal * Harare, Zimbabwe * Harare, Zimbabwe * Athens, Greece * Johannesburg, South Africa * Edmonton, Canada","label":false}
{"id":"nt-95:3","text_a":"how long was the race in the all-africa games (distance)?","text_b":"Position * 2nd * 2nd * 2nd * 7th * 3rd * 9th","label":false}
{"id":"nt-95:4","text_a":"how long was the race in the all-africa games (distance)?","text_b":"Notes * 5,000 m * 5,000 m * 10,000 m * 10,000 m * 10,000 m * 10,000 m","label":true}
```

Rows:
```json
{"id":"nt-95:0","text_a":"how long was the race in the all-africa games (distance)?","text_b":"Year : 1994 * Competition : World Junior Championships * Venue : Lisbon, Portugal * Position : 2nd * Notes : 5,000 m","label":false}
{"id":"nt-95:1","text_a":"how long was the race in the all-africa games (distance)?","text_b":"Year : 1995 * Competition : All-Africa Games * Venue : Harare, Zimbabwe * Position : 2nd * Notes : 5,000 m","label":false}
{"id":"nt-95:2","text_a":"how long was the race in the all-africa games (distance)?","text_b":"Year : 1995 * Competition : All-Africa Games * Venue : Harare, Zimbabwe * Position : 2nd * Notes : 10,000 m","label":true}
{"id":"nt-95:3","text_a":"how long was the race in the all-africa games (distance)?","text_b":"Year : 1997 * Competition : World Championships * Venue : Athens, Greece * Position : 7th * Notes : 10,000 m","label":true}
{"id":"nt-95:4","text_a":"how long was the race in the all-africa games (distance)?","text_b":"Year : 1999 * Competition : All-Africa Games * Venue : Johannesburg, South Africa * Position : 3rd * Notes : 10,000 m","label":true}
{"id":"nt-95:5","text_a":"how long was the race in the all-africa games (distance)?","text_b":"Year : 2001 * Competition : World Championships * Venue : Edmonton, Canada * Position : 9th * Notes : 10,000 m","label":true}
```

Note that in the case of WikiTableQuestions we do not have the row ids of the correct answer, so all rows with the correct answer in the correct column are considered correct.

We then train two models: one to classify question-column pairs and one to classify question-row pairs.

These models are then applied and combined to give probabilities per-cell.
