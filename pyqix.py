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

BLACK = 0
WHITE = 1

screen = None
window_surface = None
frame_counter = 0  # counts the frames (updating qix not every frame)
logo = None
color = [(0, 0, 0), (255, 255, 255), (73, 73, 73),        # BLACK,   WHITE,  DARKGREY
         (255, 255, 73), (217, 92, 74), (128, 128, 128),  # YELLOW,  MIDRED, GREY
         (255, 73, 73), (73, 73, 255), (73, 255, 73),     # RED,     BLUE,   GREEN,
         (145, 36, 18), (0, 127, 127)]                    # DARKRED, CYAN
# ------- per player --------
playfield = [[], []]  # the array of vertex, holding the borders of the game
# ---------------------
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


def draw_list(p_list, arg_color, closed=True):
    upper_limit = len(p_list)
    if not closed:
        upper_limit -= 1
    for index in range(0, upper_limit):
        hal_draw_line(p_list[index],
                      p_list[(index + 1) % len(p_list)], arg_color)


def paint_playfield():
    draw_list(playfield[0], color[WHITE])


def paint_game():
    hal_blt(logo, (24, 16))
    paint_playfield()


def reset_playfield(index_player):
    global playfield
    playfield[index_player] = new_playfield


def init():
    global window_surface, screen, logo
    pygame.init()
    window_surface = pygame.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])
    screen = pygame.Surface((WIDTH, HEIGHT))
    logo, _ = hal_load_image(os.path.join('data', 'qix_logo.png'))
    logo = pygame.transform.scale(logo, (int(56.0 * X_SCALE), int(20 * Y_SCALE)))
    reset_playfield(0)


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
