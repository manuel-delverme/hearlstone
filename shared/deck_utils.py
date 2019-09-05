import time
import os
import random
import collections
import csv

try:
    import requests
    import bs4
except ImportError:
    raise ImportError("Download bs4 and requests")

import tqdm
import hearthstone.deckstrings
import hearthstone.enums as hs_enum
import hearthstone.cardxml as cardxml
import shared.constants as C

query = "https://www.hearthpwn.com/decks?filter-unreleased-cards=f&filter-deck-tag=5&filter-deck-type-val=8&filter-deck-type-op=4&filter-class=16&sort=-rating"
base_url = "https://www.hearthpwn.com"


def spell_minion_split(deck):
    assert isinstance(deck, hearthstone.deckstrings.Deck), "Takes as input"
    minions = set()
    spells = set()
    for (idx, _) in deck.get_dbf_id_list():
        card = db[idx]
        if card.type == hs_enum.CardType.SPELL:
            spells.add(idx)
        elif card.type == hs_enum.CardType.MINION:
            minions.add(idx)
        else:
            print("card not found", card)
    return minions, spells


def get_no_power_cards(deck):
    """Takes a deck, returns the set of card_id that have spell_power."""
    assert isinstance(deck, hearthstone.deckstrings.Deck), "Takes as input"
    cards_no_power = set()
    for (idx, _) in deck.get_dbf_id_list():
        card = db[idx]
        if card.spell_damage == 0:
            continue
        else:
            cards_no_power.add(idx)
    print(f"Found {len(cards_no_power)}")
    return cards_no_power


def get_deck_string(query):
    response = requests.get(query)
    response.raise_for_status()
    response = bs4.BeautifulSoup(response.text, "html.parser")
    attr = {"data-ga-click-event-tracking-category": "Deck Copy"}
    deck_string = response.find('button', attrs=attr).get('data-clipboard-text')
    return deck_string


def get_deck_list(query, max_decks=50):
    print(f"Downloading deck from {base_url}")
    response = requests.get(query)
    response.raise_for_status()
    response = bs4.BeautifulSoup(response.text, "html.parser")
    deck_dict = collections.defaultdict(lambda: {})
    idx = 0
    deck_list = []
    for link in response.find_all('a'):
        href = link.get('href')
        if href is not None and "/decks/" in href:
            print(f"Found {href}")

            deck_address = base_url + href
            deck_string = get_deck_string(deck_address)
            deck_dict[idx] = {
                'deck_address': deck_address,
                'deck_string': deck_string
            }
            deck_list.append(deck_string)
            idx += 1
            time.sleep(5)
            print(f"Decks so far {idx}")
            if idx == max_decks:
                break
    if deck_list:
        save_decks(deck_dict)
    return deck_list


def save_decks(deck_dict):
    assert isinstance(deck_dict, dict)
    with open('decks.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['deck_address', 'deck_string'])
        writer.writeheader()
        writer.writerows(deck_dict.values())


def load_decks():
    assert os.path.exists('decks.csv'), FileNotFoundError
    decks = []
    with open('decks.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            decks.append(row['deck_string'])
    return decks


def get_decks():
    decks = None
    try:
        decks = load_decks()
    except FileNotFoundError:
        print("Couldn't load decks. File not found.")

    if decks is None:
        try:
            decks = get_deck_list(query)
        except Excetion as e:
            print(e)
        finally:
            decks = [C.DECK1]
    assert decks is not None
    return decks


def sample_deck():
    decks = get_decks()
    deck_string = random.choice(decks)
    deck = hearthstone.deckstrings.Deck().from_deckstring(deck_string)
    minions, spells = spell_minion_split(deck)

    return deck_string, minions, spells


def test_download_deck():
    deck_list = get_deck_list(query, max_decks=2)
    assert isinstance(deck_list, list)
    assert len(deck_list) == 2


def test_power_cards():
    deck = C.deck
    cards_no_power = get_no_power_cards(deck)
    for idx in cards_no_power:
        print(f"card_id={idx}\tcard_name={db[idx].name}")
    assert cards_no_power == set((672, 995, 525))


def test_minion_spell_split():
    deck = C.deck
    minions, spells = spell_minion_split(deck)
    all_ids = minions.union(spells)
    card_idx = set([card[0] for card in deck.get_dbf_id_list()])
    assert card_idx == all_ids
    print(minions, spells)


def test_db():
    global db
    del db
    try:
        assert "db" in globals(), "db not loaded"
    except AssertionError as e:
        print(e)


if "db" not in globals():
    print("Loading db")
    db, _ = cardxml.load_dbf()  # load cards using entity ids

if __name__ == "__main__":

    test_download_deck()
    #test_minion_spell_split()
    #test_power_cards()
    #test_db()
