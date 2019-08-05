import curses
import enum

import shared.constants as C


class Players(enum.Enum):
  AGENT = 0
  OPPONENT = 1
  LOG = 2


CARD_WIDTH = 1  # width of box to draw card in
CARD_HEIGHT = 3


class GUI:
  def __init__(self):
    self.screen = curses.initscr()
    self.screen.immedok(True)
    _height, _width = self.screen.getmaxyx()
    self.screen.resize(max(50, _height), _width)
    (self.game_height, self.game_width) = self.screen.getmaxyx()

    opponent_rows = (CARD_HEIGHT + 2) * 1 + 2
    player_rows = (CARD_HEIGHT + 2) * 2 + 2
    log_rows = int(self.game_height - (opponent_rows + player_rows))
    assert log_rows >= 3

    opponent_bottom = opponent_rows
    player_bottom = opponent_rows + player_rows

    opponent_top = 0
    player_top = opponent_bottom
    log_top = player_bottom

    self.windows = {
      Players.OPPONENT: curses.newwin(opponent_rows, self.game_width - 1, opponent_top, 0),
      Players.AGENT: curses.newwin(player_rows, self.game_width - 1, player_top, 0),
      Players.LOG: curses.newwin(log_rows, self.game_width - 1, log_top, 0),
    }
    for k in self.windows:
      self.windows[k].immedok(True)
      self.windows[k].box()

  def __del__(self):
    curses.endwin()

  def get_input(self, message=""):
    self.player_addstr(-3, 2, " " * (len(message) + 10))  # blank out line
    self.player_addstr(-3, 2, message)  # ,curses.A_REVERSE)
    return self.windows["players"].getstr()

  def get_key(self, message=""):
    self.player_addstr(-3, 2, " " * (len(message) + 10))  # blank out line
    self.player_addstr(-3, 2, message)  # ,curses.A_REVERSE)
    return chr(self.windows["players"].getch())

  def player_addstr(self, y, x, message):
    self.windows[Players.AGENT].addstr(y, x, message)

  def draw_rectangle(self, window, y, x, height, width):
    self.windows[window].vline(y + 1, x, curses.ACS_VLINE, height)
    self.windows[window].vline(y + 1, x + width + 1, curses.ACS_VLINE, height)
    self.windows[window].hline(y, x + 1, curses.ACS_HLINE, width)
    self.windows[window].hline(y + height + 1, x + 1, curses.ACS_HLINE, width)
    for i in range(1, height):
      self.windows[window].addstr(y + i, x + 1, " " * width)
    self.windows[window].addch(y, x, curses.ACS_ULCORNER)
    self.windows[window].addch(y, x + width + 1, curses.ACS_URCORNER)
    self.windows[window].addch(y + height + 1, x, curses.ACS_LLCORNER)
    self.windows[window].addch(y + height + 1, x + width + 1, curses.ACS_LRCORNER)

  def opponent_addstr(self, y, x, message, options=0):
    self.windows[Players.OPPONENT].addstr(y, x, message, options)

  def draw_opponent(self, hero, board, hand, mana):
    top_row = [C.Minion(atk=hero.health, health=hero.power_exhausted, exhausted=hero.atk_exhausted)]
    top_row.extend(C.Minion(minion.health, minion.atk, minion.exhausted) for minion in board)
    self.draw_player_side(Players.OPPONENT, top_row=None, bottom_row=top_row)
    nametag = 'Opponent mana:{}'.format(mana)
    self.opponent_addstr(0, self.game_width - 1 - len(nametag), nametag)

  def draw_agent(self, hero, board, hand, mana):
    top_row = [Minion(atk=hero.power_exhausted, health=hero.health, exhausted=hero.atk_exhausted)]
    top_row.extend(Minion(minion.atk, minion.health, minion.exhausted) for minion in board)

    self.draw_player_side(Players.AGENT, top_row=top_row, bottom_row=hand)
    nametag = 'Agent mana:{}'.format(mana)
    self.player_addstr(0, self.game_width - 1 - len(nametag), nametag)

  def draw_player_side(self, player, top_row, bottom_row):
    self.windows[player].clear()
    self.windows[player].box()
    offset = 1
    if top_row:
      self.draw_zone(top_row, player, offset_row=offset, offset_column=1)
      offset += CARD_HEIGHT + 2
    self.draw_zone(bottom_row, player, offset_row=offset, offset_column=1)

  def draw_zone(self, cards_to_draw, player, offset_column, offset_row):
    for offset, card in enumerate(cards_to_draw):
      assert isinstance(card, (Minion, Hero, Card))
      pixel_offset = offset * (CARD_WIDTH + 4)

      ready = '+'
      if isinstance(card, Minion):
        if card.exhausted:
          ready = 'z'
      elif isinstance(card, Card) and not card.health:
        card_id = ''.join(i for i in str(SPELLS(card.id))[7:] if i.isupper())
        if len(card_id) == 1:
          card_id = str(SPELLS(card.id))[7:9]
        card = Card(atk=card_id, health=card.cost, id=None, cost=None)
        ready = 's'

      self.draw_rectangle(player, offset_row, offset_column + pixel_offset, CARD_HEIGHT, CARD_WIDTH)
      self.windows[player].addstr(offset_row + 0, offset_column + 2 + pixel_offset, str(ready))
      self.windows[player].addstr(offset_row + 1, offset_column + 1 + pixel_offset, str(card.atk))
      self.windows[player].addstr(offset_row + CARD_HEIGHT,
                                  offset_column + 1 + CARD_WIDTH - len(str(card.health)) + pixel_offset,
                                  str(card.health))

  def log(self, txt, row=1, multiline=False):
    self.windows[Players.LOG].addstr(row, 1, txt)

    if multiline:
      self.windows[Players.LOG].clrtobot()  # clear the rest of the line
    else:
      self.windows[Players.LOG].clrtoeol()  # clear the rest of the line
    self.windows[Players.LOG].box()
