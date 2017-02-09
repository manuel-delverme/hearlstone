import fireplace
import random
from hearthstone.enums import CardClass, CardType

from fireplace.game import Game
from fireplace.player import Player

def setup_game() -> ".game.Game":
    deck1 = fireplace.utils.random_draft(CardClass.MAGE)
    deck2 = fireplace.utils.random_draft(CardClass.WARRIOR)
    player1 = Player("Player1", deck1, CardClass.MAGE.default_hero)
    player2 = Player("Player2", deck2, CardClass.WARRIOR.default_hero)

    game = Game(players=(player1, player2))
    game.start()

    return game

class Action(object):
    def __init__(self, card, usage, params):
        self.card = card
        self.usage = usage
        self.params = params
    def use(self):
        self.usage(**self.params)

class HSsimulation(object):
    def __init__(self):
        game = setup_game()

        self.game = game
        self.player = self.game.players[0]
        self.opponent = self.game.players[1]

    def terminal(self):
        return self.game.ended

    def reward(self):
        score = len(self.player.hand)
        score -= len(self.opponent.hand)

        score += self.player.hero.health
        score -= self.opponent.hero.health

        for entity in self.game.board:
            if entity.controller == self.player:
                sign = +1
            else:
                sign = -1
            score += sign * entity.atk
            score += sign * entity.health
        return score

    def actions(self):
        actions = []

        for card in self.player.hand:
            if card.is_playable():
                # if card.must_choose_one:
                #     card = random.choice(card.choose_cards)

                if card.requires_target():
                    for target in card.targets:
                        actions.append(Action(card, card.play, {'target':target}))
                else:
                    actions.append(Action(card, card.play, {'target':None}))

        for character in self.player.characters:
            if character.can_attack():
                for enemy_char in character.targets:
                    actions.append(Action(card, card.play, {'target':None}))
                    actions.append(Action(card, character.attack, {'target':enemy_char}))
        return actions

    def all_actions(self):
        actions = []

        for card in self.player.hand:
            if card.requires_target():
                for target in card.targets:
                    actions.append(lambda : card.play(target=target))
            else:
                actions.append(lambda : card.play(target=None))

        for character in self.player.characters:
            for enemy_char in character.targets:
                actions.append(lambda :character.attack(enemy_char))
        return actions

    def card_to_vector(self, card):
        try:
            SPELL_TYPE = 5
            WEAPON_TYPE = 7
            if card.type == SPELL_TYPE:
                features = [card.cost, 0, 0]
            elif card.type == WEAPON_TYPE:
                features = [card.cost, card.durability, card.atk]
            else:
                features = [card.cost, card.health, card.atk]
        except Exception as e:
            import ipdb; ipdb.set_trace()
            print("warning", e)
        return features

    def observe_player(self, player):
        state = []
        for card in player.hand:
            state.extend(self.card_to_vector(card))
        for character in player.characters:
            state.extend(self.card_to_vector(card))
        state.append(player.mana)
        return state

    def observe(self):
        state = self.observe_player(self.player)
        state.extend(self.observe_player(self.opponent))
        return state # np.array(state)

    def step(self, action):
        observation = self.observe()
        action.use()
        next_observation = self.observe()
        reward = self.reward()
        terminal = self.terminal()
        return (observation, action, next_observation, reward, terminal)

if __name__ == "__main__":
    fireplace.cards.db.initialize()
    sim1 = HSsimulation()

    for player in sim1.game.players:
        print("Can mulligan %r" % (player.choice.cards))
        mull_count = random.randint(0, len(player.choice.cards))
        cards_to_mulligan = random.sample(player.choice.cards, mull_count)
        player.choice.choose(*cards_to_mulligan)

    for i in range(10):
        fireplace.utils.play_turn(sim1.game)
        actions = sim1.actions()
        if len(actions) > 0:
            break

    print("terminal", sim1.terminal())
    print("reward", sim1.reward())
    actions = sim1.actions()
    print("actions", actions)
    print("all_actions", sim1.all_actions())
    print("observe", sim1.observe())
    print("acts:", actions[0])
    observation, action, next_observation, reward, terminal = sim1.step(actions[0])
    print("observation+", observation)
    print("action+", action)
    print("next_observation+", next_observation)
    print("reward+", reward)
    print("terminal+", terminal)
