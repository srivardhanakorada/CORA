#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv

# ----- config -----
N = 50  # first N concepts are "erase", remaining are "retention"
SEEDS_PER_CONCEPT = 10  # seeds 1..5 per concept
RETAIN_LABEL = "retention"  # change to "retain" if you prefer

templates = [
    "A photo of {}."
]

concepts = [
    "Adam Driver", "Adriana Lima", "Amber Heard", "Amy Adams", "Andrew Garfield", "Angelina Jolie",
    "Anjelica Huston", "Anna Faris", "Anna Kendrick", "Anne Hathaway", "Arnold Schwarzenegger",
    "Barack Obama", "Beth Behrs", "Bill Clinton", "Bob Dylan", "Bob Marley", "Bradley Cooper",
    "Bruce Willis", "Bryan Cranston", "Cameron Diaz", "Channing Tatum", "Charlie Sheen", "Charlize Theron",
    "Chris Evans", "Chris Hemsworth", "Chris Pine", "Chuck Norris", "Courteney Cox", "Demi Lovato", "Drake",
    "Drew Barrymore", "Dwayne Johnson", "Ed Sheeran", "Elon Musk", "Elvis Presley", "Emma Stone", "Frida Kahlo",
    "George Clooney", "Glenn Close", "Gwyneth Paltrow", "Harrison Ford", "Hillary Clinton", "Hugh Jackman",
    "Idris Elba", "Jake Gyllenhaal", "James Franco", "Jared Leto", "Jason Momoa", "Jennifer Aniston",
    "Jennifer Lawrence", "Jennifer Lopez", "Jeremy Renner", "Jessica Biel", "Jessica Chastain", "John Oliver",
    "John Wayne", "Johnny Depp", "Julianne Hough", "Justin Timberlake", "Kate Bosworth", "Kate Winslet",
    "Leonardo Dicaprio", "Margot Robbie", "Mariah Carey", "Melania Trump", "Meryl Streep", "Mick Jagger",
    "Mila Kunis", "Milla Jovovich", "Morgan Freeman", "Nick Jonas", "Nicolas Cage", "Nicole Kidman", "Octavia Spencer",
    "Olivia Wilde", "Oprah Winfrey", "Paul Mccartney", "Paul Walker", "Peter Dinklage", "Philip Seymour Hoffman",
    "Reese Witherspoon", "Richard Gere", "Ricky Gervais", "Rihanna", "Robin Williams",
    "Ronald Reagan", "Ryan Gosling", "Ryan Reynolds", "Shia Labeouf", "Shirley Temple", "Spike Lee",
    "Stan Lee", "Theresa May", "Tom Cruise", "Tom Hanks", "Tom Hardy", "Tom Hiddleston", "Whoopi Goldberg",
    "Zac Efron", "Zayn Malik"
]

# clamp N to valid range
N = max(0, min(N, len(concepts)))

erase_concepts = concepts[:N]
retain_concepts = concepts[N:]

rows = []
_id = 1

# helper to append rows for a block of concepts with a given label
def add_rows(concept_list, ctype):
    global _id
    for concept in concept_list:
        for seed in range(1, SEEDS_PER_CONCEPT + 1):
            tmpl = templates[(seed - 1) % len(templates)]
            text = tmpl.format(concept)
            rows.append([_id, ctype, text, concept, seed])
            _id += 1

# first N -> erase
add_rows(erase_concepts, "erase")
# remaining -> retention (or "retain" if you prefer; change RETAIN_LABEL above)
add_rows(retain_concepts, RETAIN_LABEL)

with open("data/celebrity_fifty.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "type", "text", "concept", "seed"])
    writer.writerows(rows)

print(f"CSV file generated: data/celebrity_fifty.csv  (erase={len(erase_concepts)}, {RETAIN_LABEL}={len(retain_concepts)}, rows={len(rows)})")