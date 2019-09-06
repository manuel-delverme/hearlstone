import time
import os
import random
import collections
import csv
import warnings

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

QUERY = "https://www.hearthpwn.com/decks?filter-unreleased-cards=f&filter-deck-tag=5&filter-deck-type-val=8&filter-deck-type-op=4&filter-class=16&sort=-rating"
BASE_URL = "https://www.hearthpwn.com"


def spell_minion_split(deck):
    assert isinstance(deck, hearthstone.deckstrings.Deck
                     ), "deck is not instance of hearthstone.deckstrings.Deck"
    minions = set()
    spells = set()
    for (idx, _) in deck.get_dbf_id_list():
        try:
            card = db[idx]
        except KeyError:
            raise KeyError(f"card {idx} not in hearthstone db")
        if card.type == hs_enum.CardType.SPELL:
            spells.add(idx)
        elif card.type == hs_enum.CardType.MINION:
            minions.add(idx)
        else:
            raise ValueError(f"Card {idx, card.type} is not a spell or minion")
    return minions, spells


def get_no_spell_power_cards(deck):
    """Takes a deck, returns the set of card_id with no spell_power."""
    assert isinstance(deck, hearthstone.deckstrings.Deck
                     ), "deck is not instance of hearthstone.deckstrings.Deck"
    no_spell_power_cards = set()
    for (idx, _) in deck.get_dbf_id_list():
        card = db[idx]
        if card.spell_damage == 0:
            continue
        else:
            no_spell_power_cards.add(idx)
    print(f"Found {len(no_spell_power_cards)}")
    return no_spell_power_cards


def get_deck_string(query):
    response = requests.get(query)
    response.raise_for_status()
    response = bs4.BeautifulSoup(response.text, "html.parser")
    attr = {"data-ga-click-event-tracking-category": "Deck Copy"}
    deck_string = response.find('button', attrs=attr).get('data-clipboard-text')
    return deck_string


def get_deck_list(query, max_decks=50):
    print(f"Downloading deck from {BASE_URL}")
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

            deck_address = BASE_URL + href
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
        deck_dict = check_decks(deck_dict)
        save_decks(deck_dict)
    return deck_list


def check_decks(decks):
    valid_decks = {}
    for idx, _deck in decks.items():
        deck = hearthstone.deckstrings.Deck().from_deckstring(
            _deck['deck_string'])
        try:
            spell_minion_split(deck)
            valid_decks[idx] = _deck
        except (KeyError, ValueError):
            print(f"Removing deck {_deck['deck_address']}")
    assert len(valid_decks.keys()), "No valid decks found."
    return valid_decks


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
    decks = [C.DECK1]
    if os.path.exists('decks.csv'):
        decks = load_decks()
    else:
        try:
            decks = get_deck_list(QUERY)
        except Excetion as e:
            warnings.warn(
                "Could not download deck list. Game starts with default deck.")
    return decks


def sample_deck(decks=None):
    if decks is None:
        decks = get_decks()
    deck_string = random.choice(decks)
    deck = hearthstone.deckstrings.Deck().from_deckstring(deck_string)
    minions, spells = spell_minion_split(deck)
    return deck_string, minions, spells


if "db" not in globals():
    print("Loading card db")
    db, _ = cardxml.load_dbf()  # load cards using entity ids

if __name__ == "__main__":
    get_deck_list(QUERY)
