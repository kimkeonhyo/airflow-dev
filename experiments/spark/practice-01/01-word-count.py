from collections import defaultdict

if __name__ == '__main__':
    word_count: dict[str, int] = defaultdict(int)

    with open("data/spark/LoremIpsum.txt", "r") as file:
        for _, line in enumerate(file):
            words_splitted_space: list[str] = line.strip().split()
            for _, word in enumerate(words_splitted_space):
                word_count[word] += 1

    for word, count in word_count.items():
        print(f"{word}: {count}")