# !/usr/bin/env python
import time
import curses
import random
import enum


class Players(enum.Enum):
  AGENT = 0
  OPPONENT = 1
  LOG = 2


CARD_WIDTH = 1  # width of box to draw card in
CARD_HEIGHT = 2


class GUI:
  def __init__(self):
    self.screen = curses.initscr()
    self.screen.immedok(True)
    (self.game_height, self.game_width) = self.screen.getmaxyx()
    # if self.game_height < 60 or self.game_width < 180:
    # print "\x1b[8;60;180t" #resize terminal
    # (self.game_height, self.game_width) = self.screen.getmaxyx()
    # (self.game_height, self.game_width) = (58, 180)

    opponent_rows = int(self.game_height / 3)
    player_rows = int(self.game_height / 3)
    log_rows = int(self.game_height / 3)

    opponent_bottom = int(self.game_height / 3)
    player_bottom = int(self.game_height * 2 / 3)
    log_bottom = int(self.game_height)

    opponent_top = 0
    player_top = opponent_bottom + 1
    log_top = player_bottom + 1

    self.windows = {
      Players.OPPONENT: curses.newwin(opponent_rows, self.game_width - 1, opponent_top, 0),
      Players.AGENT: curses.newwin(player_rows, self.game_width - 1, player_top, 0),
      Players.LOG: curses.newwin(log_rows, self.game_width - 1, log_top, 0),
    }
    for k in self.windows:
        self.windows[k].immedok(True)
        self.windows[k].box()
        # print('window:', k)
        time.sleep(1)

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

  def draw_player_side(self, player, board_cards, hand_cards):
    self.draw_zone(board_cards, player, offset_row=1, offset_column=1)
    self.draw_zone(hand_cards, player, offset_row=CARD_HEIGHT + 3, offset_column=1)

  def draw_zone(self, cards_to_draw, player, offset_column, offset_row):
    for offset, card in enumerate(cards_to_draw):
      pixel_offset = offset * (CARD_WIDTH + 4)
      self.draw_rectangle(player, offset_row, offset_column + pixel_offset, CARD_HEIGHT, CARD_WIDTH)
      self.windows[player].addstr(offset_row + 1, offset_column + 1 + pixel_offset, str(card))
      self.windows[player].addstr(offset_row + CARD_HEIGHT, offset_column + 1 + CARD_WIDTH - len(str(card)) + pixel_offset, str(card))


if __name__ == "__main__":
  gui = GUI()

  card_value = 3
  gui.opponent_addstr(0, 0, 'opponent')
  gui.player_addstr(0, 0, 'player')
  # if self.revealed:
  #   gui.opponent_addstr(y + 2, x - 15, "Card value: " + card_value)
  # else:
  #   gui.opponent_addstr(y + 2, x - 15, "Cards:")
  def log(txt):
      time.sleep(1)
      gui.windows[Players.LOG].addstr( 1,  1 , txt)

  board = range(7)
  hand = range(10)
  log('draw1')
  gui.draw_player_side(Players.OPPONENT, board, hand)
  log('draw2')
  gui.draw_player_side(Players.AGENT, board, hand)
  for i in range(5):
    log('sleep %s' % i)
  # print('sleep')

  # g = Game()
  # while g.play():
  del gui
  # print("Played a total of " + str(round_count) + " rounds.")
