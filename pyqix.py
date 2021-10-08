import os

import pygame
from pygame.locals import *

WIDTH = 1024
HEIGHT = 1024
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 1024
GAME_WIDTH = 256.0  # internal game logic resolution
GAME_HEIGHT = 256.0
X_SCALE = (WIDTH / GAME_WIDTH)
Y_SCALE = (HEIGHT / GAME_HEIGHT)

TPS = 60  # number of game updates per second
SKIP_TICKS = 1000 / TPS  # ms to start skipping frames
MAX_FRAMESKIP = 5  # no we calc max updates (if we are behind) before displaying
MAX_TIMESKIP = 2000  # max Time we try to catch up until we just reset counter

BLACK = FONT_NORMAL = 0
WHITE = CENTER_X = FONT_LARGE = 1
DARKGREY = CENTER_Y = FONT_SCORE = 2
YELLOW = CENTER = 3
MIDRED = NO_BLIT = 4
RED = 6

fonts = None
screen = None
window_surface = None
logo = None
active_live = None
inactive_live = None
color = [(0, 0, 0), (255, 255, 255), (73, 73, 73),        # BLACK,   WHITE,  DARKGREY
         (255, 255, 73), (217, 92, 74), (128, 128, 128),  # YELLOW,  MIDRED, GREY
         (255, 73, 73), (73, 73, 255), (73, 255, 73),     # RED,     BLUE,   GREEN,
         (145, 36, 18), (0, 127, 127)]                    # DARKRED, CYAN
frame_counter = 0  # counts the frames (updating qix not every frame)
current_player = 0
max_player = 2
# ------- per player --------
playfield = [[], []]  # the array of vertex, holding the borders of the game
player_coords = [[], []]  # an  x/y coordinate
player_lives = [0, 0]  # num lives of both players
scores = [0, 0]
# ---------------------
player_size = 3.0
player_start = [128, 239]
start_player_lives = 3
live_coord = (234, 14)
highscore = [(30000, "QIX") for i in range(10)]
new_playfield = [(16, 39), (16, 239), (240, 239), (240, 39)]


def hal_blt(img, coords):
    screen.blit(img, (coords[0] * X_SCALE, coords[1] * Y_SCALE))


def hal_load_image(fullname, color_key=None):
    try:
        image = pygame.image.load(fullname)
    except pygame.error as message:
        print('Cannot load image:', fullname)
        raise SystemExit(message)
    image = image.convert()
    if color_key is not None:
        if color_key == -1:
            color_key = image.get_at((0, 0))
        image.set_colorkey(color_key, RLEACCEL)
    return image, image.get_rect()


def hal_draw_line(point_1, point_2, arg_color):
    point_1 = (point_1[0] * X_SCALE, point_1[1] * Y_SCALE)
    point_2 = (point_2[0] * X_SCALE, point_2[1] * Y_SCALE)
    pygame.draw.line(screen, arg_color, point_1, point_2)


def get_real_time():
    """ Real time of the system in ms after pygame.init"""
    return pygame.time.get_ticks()


def hal_draw_rect(point_1, point_2, arg_color):
    point_1 = (point_1[0] * X_SCALE, point_1[1] * Y_SCALE)
    point_2 = (point_2[0] * X_SCALE, point_2[1] * Y_SCALE)
    pygame.draw.rect(screen, arg_color, (point_1[0], point_1[1], point_2[0] - point_1[0], point_2[1] - point_1[1]))


def print_at(str_text, coords, txt_color=color[YELLOW], center_flags=0, anti_aliasing=1, use_font=FONT_NORMAL):
    if center_flags & 0x04 == 0:
        text = fonts[use_font].render(str_text, anti_aliasing, txt_color)
    else:
        return fonts[use_font].size(str_text)
    xco = coords[0]
    yco = coords[1]
    w = text.get_rect().width
    h = text.get_rect().height
    if center_flags & 0x01 != 0:
        xco = (WIDTH - w) / 2 / X_SCALE
    if center_flags & 0x02 != 0:
        yco = (HEIGHT - h) / 2 / Y_SCALE
    text_pos = Rect(xco, yco, w, h)
    hal_blt(text, text_pos)
    return text_pos[2:]


def vector_add(v1, v2):
    return [v1[0] + v2[0], v1[1] + v2[1]]


def vector_sub(v1, v2):
    return [v1[0] - v2[0], v1[1] - v2[1]]


def draw_list(p_list, arg_color, closed=True):
    upper_limit = len(p_list)
    if not closed:
        upper_limit -= 1
    for index in range(0, upper_limit):
        hal_draw_line(p_list[index],
                      p_list[(index + 1) % len(p_list)], arg_color)


def paint_score():
    print_at("%d  %s" % (highscore[0][0], highscore[0][1]), (0, 13),
             txt_color=color[YELLOW], center_flags=CENTER_X, anti_aliasing=0)
    for index in range(max_player):      # paint score for both player
        dim = print_at(str(scores[index]), (0, 0), color[WHITE], center_flags=NO_BLIT, use_font=FONT_SCORE)
        coords = (232 - dim[0] / X_SCALE, 16 + index * 11)
        print_at(str(scores[index]), coords, color[WHITE], use_font=FONT_SCORE)


def paint_claimed_and_lives():
    print_at("CLAIMED", (0, 22), txt_color=color[YELLOW], center_flags=CENTER_X, anti_aliasing=0)
    print_at("0%  75%", (0, 29), txt_color=color[YELLOW], center_flags=CENTER_X, anti_aliasing=0)
    for player in range(max_player):
        for index in range(start_player_lives):
            start_coord = [(live_coord[0] + (index // 3) * 4),
                           (live_coord[1] + 11 * current_player + (index % 3) * 4)]
            if player_lives[player] == start_player_lives - index:
                hal_blt(active_live, start_coord)
            else:
                hal_blt(inactive_live, start_coord)


def paint_player():
    pos = player_coords[current_player]
    player = [vector_add(pos, (-player_size, 0)), vector_add(pos, (0, player_size)),
              vector_add(pos, (player_size, 0)), vector_add(pos, (0, -player_size))]
    draw_list(player, color[RED], True)
    if X_SCALE > 1.0:
        hal_draw_rect(vector_add(pos, (-1, -1)), vector_add(pos, (1, 1)), color[WHITE])
    else:
        hal_draw_rect(pos, vector_add(pos, (1, 1)), color[WHITE])  # add single pixel (pygame draws a 3x3 rect on w=0)


def paint_playfield():
    draw_list(playfield[current_player], color[WHITE])


def paint_game():
    hal_blt(logo, (24, 16))
    paint_score()
    paint_playfield()
    paint_claimed_and_lives()
    paint_player()


def reset_playfield(index_player):
    global playfield
    playfield[index_player] = new_playfield


def init():
    global window_surface, screen, logo, fonts, active_live, inactive_live, player_lives, player_coords
    pygame.init()
    window_surface = pygame.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])
    screen = pygame.Surface((WIDTH, HEIGHT))
    fonts = [pygame.font.Font("data/qix-small.ttf", int(8.0 * Y_SCALE)),
             pygame.font.Font("data/qix-large.ttf", int(8.0 * Y_SCALE)),
             pygame.font.Font("data/qix-large.ttf", int(10.0 * Y_SCALE))]
    logo, _ = hal_load_image(os.path.join('data', 'qix_logo.png'))
    logo = pygame.transform.scale(logo, (int(56.0 * X_SCALE), int(20 * Y_SCALE)))
    active_live, _ = hal_load_image(os.path.join('data', 'qix_live_w.png'))
    active_live = pygame.transform.scale(active_live, (int(3.0 * X_SCALE), int(3.0 * Y_SCALE)))
    inactive_live, _ = hal_load_image(os.path.join('data', 'qix_live_r.png'))
    inactive_live = pygame.transform.scale(inactive_live,  (int(3.0 * X_SCALE), int(3.0 * Y_SCALE)))
    reset_playfield(0)
    player_lives = [start_player_lives, start_player_lives]
    player_coords = [player_start, player_start]


def gameloop():  # https://dewitters.com/dewitters-gameloop/
    global frame_counter
    next_game_tick = get_real_time() - 1
    while 1:
        loops = 0
        while get_real_time() > next_game_tick and loops < MAX_FRAMESKIP:
            for event in pygame.event.get():
                if event.type == QUIT:
                    return
            frame_counter += 1
            next_game_tick += SKIP_TICKS
            if get_real_time() > next_game_tick + MAX_TIMESKIP:
                next_game_tick = get_real_time() + SKIP_TICKS
            loops += 1
        screen.fill(color[BLACK])
        paint_game()
        window_tmp = pygame.transform.scale(screen, (WINDOW_WIDTH, WINDOW_HEIGHT))
        window_surface.blit(window_tmp, (0, 0))
        pygame.display.flip()


if __name__ == '__main__':
    print("pyQix")
    print("-----")
    print("(c) 2021 YogaSurfTech https://github.com/YogaSurfTech/pyqix")
    print("A faithful remake of the Taito classic arcade game QIX in Python. ")
    print("On the occasion of the 40th anniversary of the release at 18th. October 2021")
    init()
    gameloop()
