import csv
imagenet_templates = [
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}."
]
concept_groups = {
    "erase": ["Snoopy", "Mickey", "Spongebob"],
    "retention": [ "Pikachu", "Dog", "Legislator"],
}
N = len(imagenet_templates)
rows = []
_id = 1
for ctype, concepts in concept_groups.items():
    for concept in concepts:
        for seed in range(1, N + 1):
            template = imagenet_templates[(seed - 1) % len(imagenet_templates)]
            text = template.format(concept)
            rows.append([_id, ctype, text, concept, seed])
            _id += 1
with open("concept_prompts.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "type", "text", "concept", "seed"])
    writer.writerows(rows)
print("CSV file generated: concept_prompts.csv")